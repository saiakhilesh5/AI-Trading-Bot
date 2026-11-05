"""Paper-execution integration tests (live dev DB, cleaned up per test)."""
from __future__ import annotations

import pytest

from core.interfaces import DecisionAction, TradeDecision
from data_layer.storage.data_access import get_session, init_database
from data_layer.storage.schemas import Position as PositionORM
from data_layer.storage.schemas import Fill, Order, PortfolioState
from execution.order_manager import OrderManager
from execution.paper_broker import PaperBroker
from execution.portfolio import Portfolio
from execution.reconciliation import sync_open_positions
from tests.risk.test_risk_engine import make_trade


@pytest.fixture()
def clean_portfolio_db():
    init_database()
    with get_session() as session:
        session.query(PositionORM).delete()
        session.query(PortfolioState).delete()
        session.query(Fill).delete()
        session.query(Order).delete()
        session.commit()
    yield
    with get_session() as session:
        session.query(PositionORM).delete()
        session.query(PortfolioState).delete()
        session.query(Fill).delete()
        session.query(Order).delete()
        session.commit()


def make_decision(action=DecisionAction.BUY, notional=20000.0) -> TradeDecision:
    trade = make_trade().model_copy(update={"supporting_signals":
                                            [f"risk_approved:notional={notional}"]})
    from core.clock import get_clock
    return TradeDecision(decision_id="d-test-exec-1", ts=get_clock().now(),
                         asset_symbol="XAUUSD", action=action, proposed_trade=trade)


@pytest.mark.usefixtures("clean_portfolio_db")
class TestPortfolio:
    def test_initial_capital(self):
        p = Portfolio(starting_capital=100000)
        assert p.snapshot({}) .equity == 100000
        assert p.snapshot({}).available_balance == 100000

    def test_reserve_release(self):
        p = Portfolio(starting_capital=100000)
        assert p.reserve(30000)
        assert p.snapshot({}).available_balance == pytest.approx(70000)
        p.release(30000)
        assert p.snapshot({}).available_balance == pytest.approx(100000)

    def test_insufficient_reserve(self):
        p = Portfolio(starting_capital=1000)
        assert not p.reserve(2000)

    def test_daily_pnl_and_streak(self):
        p = Portfolio(starting_capital=100000)
        p.record_realized(-100.0)
        assert p.snapshot({}).consecutive_losses == 1
        p.record_realized(-50.0)
        assert p.snapshot({}).consecutive_losses == 2
        p.record_realized(200.0)
        assert p.snapshot({}).consecutive_losses == 0
        assert p.snapshot({}).daily_realized_pnl == pytest.approx(50.0)

    def test_peak_equity_drawdown(self):
        p = Portfolio(starting_capital=100000)
        p.update_peak(105000)
        snap = p.snapshot({}, )  # equity still 100k cash
        assert snap.drawdown_pct == pytest.approx(100 * 5000 / 105000, rel=0.01)


@pytest.mark.usefixtures("clean_portfolio_db")
class TestPaperBroker:
    def test_open_position(self):
        broker = PaperBroker()
        pos = broker.execute_decision(make_decision(), ref_price=2000.0,
                                      asset_class="metals", atr=15.0)
        assert pos is not None and pos.status == "open"
        assert pos.stop_loss == 1960.0 and pos.take_profit == 2072.0
        assert pos.entry_price >= 2000.0  # buy pays spread+slippage

    def test_no_trade_decision_ignored(self):
        broker = PaperBroker()
        d = make_decision(action=DecisionAction.NO_TRADE)
        d = d.model_copy(update={"proposed_trade": None})
        assert broker.execute_decision(d, 2000.0, "metals", 15.0) is None

    def test_stop_loss_close(self):
        broker = PaperBroker()
        broker.execute_decision(make_decision(), ref_price=2000.0,
                                asset_class="metals", atr=15.0)
        closed = broker.process_marks({"XAUUSD": 1950.0})  # below SL
        assert len(closed) == 1
        with get_session() as session:
            pos = session.query(PositionORM).one()
            assert pos.status == "closed"
            assert pos.realized_pnl < 0  # stop-out is a loss by construction

    def test_take_profit_close(self):
        broker = PaperBroker()
        broker.execute_decision(make_decision(), ref_price=2000.0,
                                asset_class="metals", atr=15.0)
        closed = broker.process_marks({"XAUUSD": 2080.0})  # above TP
        assert len(closed) == 1

    def test_no_bracket_hit_stays_open(self):
        broker = PaperBroker()
        broker.execute_decision(make_decision(), ref_price=2000.0,
                                asset_class="metals", atr=15.0)
        assert broker.process_marks({"XAUUSD": 2010.0}) == []

    def test_insufficient_balance_rejected(self):
        p = Portfolio(starting_capital=1000)
        broker = PaperBroker(portfolio=p)
        with pytest.raises(Exception):
            broker.execute_decision(make_decision(notional=20000.0),
                                    ref_price=2000.0, asset_class="metals", atr=15.0)

    def test_short_position_tp(self):
        trade = make_trade(direction="short", entry=2000.0, stop=2040.0, tp=1928.0)
        trade = trade.model_copy(update={"supporting_signals":
                                        ["risk_approved:notional=10000"]})
        from core.clock import get_clock
        d = TradeDecision(decision_id="d-test-exec-2", ts=get_clock().now(),
                          asset_symbol="XAUUSD", action=DecisionAction.SELL,
                          proposed_trade=trade)
        broker = PaperBroker()
        pos = broker.execute_decision(d, ref_price=2000.0, asset_class="metals", atr=15.0)
        assert pos.side == "sell"
        assert broker.process_marks({"XAUUSD": 1920.0})  # below TP for a short


@pytest.mark.usefixtures("clean_portfolio_db")
class TestReconciliation:
    def test_clean_reconciliation(self):
        broker = PaperBroker()
        broker.execute_decision(make_decision(), ref_price=2000.0,
                                asset_class="metals", atr=15.0)
        report = broker.reconcile({"XAUUSD": 2000.0})
        assert report["anomalies"] == []
        assert report["checked"] == 1

    def test_detects_missing_stop(self):
        broker = PaperBroker()
        broker.execute_decision(make_decision(), ref_price=2000.0,
                                asset_class="metals", atr=15.0)
        with get_session() as session:
            pos = session.query(PositionORM).one()
            pos.stop_loss = None
            session.commit()
        report = sync_open_positions(broker.portfolio, {"XAUUSD": 2000.0})
        assert any("position_without_stop" in a for a in report["anomalies"])


class TestOrderFrequency:
    def test_offline_manager(self):
        om = OrderManager(redis_client=False)
        assert om.orders_today("XAUUSD") == 0

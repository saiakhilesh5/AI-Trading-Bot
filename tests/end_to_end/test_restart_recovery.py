"""Restart-recovery tests: the system must restart without losing the paper
portfolio or duplicating orders (spec requirement)."""
from __future__ import annotations

import pytest

from core.interfaces import DecisionAction, TradeDecision
from data_layer.storage.data_access import get_session, init_database
from data_layer.storage.schemas import (Fill, Order, PortfolioState,
                                        Position as PositionORM)
from execution.paper_broker import PaperBroker
from execution.portfolio import Portfolio
from tests.execution.test_execution import make_decision


@pytest.fixture()
def clean_state_db():
    init_database()
    with get_session() as session:
        for t in (PositionORM, PortfolioState, Fill, Order):
            session.query(t).delete()
        session.commit()
    yield
    with get_session() as session:
        for t in (PositionORM, PortfolioState, Fill, Order):
            session.query(t).delete()
        session.commit()


@pytest.mark.usefixtures("clean_state_db")
class TestRestartRecovery:
    def test_portfolio_survives_restart(self):
        p1 = Portfolio(starting_capital=100000)
        p1.reserve(25000)
        # "restart": a fresh instance over the same DB
        p2 = Portfolio(starting_capital=100000)
        snap = p2.snapshot({})
        assert snap.available_balance == pytest.approx(75000)
        # starting_capital must not reset cash on restart
        with get_session() as session:
            row = session.get(PortfolioState, 1)
            assert row.cash == pytest.approx(75000)
            assert row.starting_capital == 100000

    def test_open_position_survives_restart(self):
        broker1 = PaperBroker()
        broker1.execute_decision(make_decision(), ref_price=2000.0,
                                  asset_class="metals", atr=15.0)
        # "restart": a brand-new broker reads the same DB
        broker2 = PaperBroker()
        closed = broker2.process_marks({"XAUUSD": 1950.0})  # below SL
        assert len(closed) == 1  # exactly one position, closed exactly once

    def test_no_duplicate_position_after_restart(self):
        broker1 = PaperBroker()
        broker1.execute_decision(make_decision(), ref_price=2000.0,
                                  asset_class="metals", atr=15.0)
        broker2 = PaperBroker()
        with get_session() as session:
            n = session.query(PositionORM).filter(PositionORM.status == "open").count()
            assert n == 1  # still exactly one open position

    def test_replaying_same_decision_is_rejected(self):
        """Duplicate-order protection: the same decision id cannot open twice."""
        broker = PaperBroker()
        d = make_decision()
        broker.execute_decision(d, ref_price=2000.0, asset_class="metals", atr=15.0)
        with pytest.raises(Exception):
            broker.execute_decision(d, ref_price=2000.0, asset_class="metals", atr=15.0)
        with get_session() as session:
            assert session.query(PositionORM).count() == 1

    def test_cash_moves_only_once(self):
        broker = PaperBroker()
        d = make_decision()
        broker.execute_decision(d, ref_price=2000.0, asset_class="metals", atr=15.0)
        with get_session() as session:
            row = session.get(PortfolioState, 1)
            # 100k - ~20k entry notional - fill premium
            assert row.cash < 81000
            assert row.cash > 79000

    def test_loss_streak_persists(self):
        p = Portfolio(starting_capital=100000)
        p.record_realized(-100)
        p2 = Portfolio(starting_capital=100000)  # restart
        assert p2.snapshot({}).consecutive_losses == 1

DEFINITIONS
Vocabulary for the data schema, not rules. A definition alone is never a violation.
Absent numeric field = 0. If context has a field not described here, context wins.
D1 today — the only reference date.
D2 complete month — last day < today; the month containing today is partial.
D3 COMPLETE_MONTHS = monthly_summary months minus the partial one; CURRENT_PARTIAL = that one.
D4 "last month" = calendar month before today's month. Not trailing 30 days, not today's month.
D5 net_purchases(M) = purchases − refunds.
D6 total_spending(M) = net_purchases(M) + transfers_to_others. transfers_own is never spending.
D7 avg_monthly_spending = mean total_spending over COMPLETE_MONTHS.
D8 monthly_summary = source for per-month aggregates. transactions = source for merchant, category and single-operation questions; it is a sample, not the full ledger.
D9 available(acc) = balance − on_hold.
D10 own_funds = Σ balance over deposit products, zero-balance included. credit_limit / available_credit are never own funds.
D11 net_position = own_funds − Σ current_debt.
D12 ACCOUNT_ROSTER = every element of accounts, zero-balance and credit ones included.
D13 liquid = checking + flexible savings − on_hold; Ahorro Fijo before maturity_date is not liquid.
D14 applicable_rate = account.annual_rate, else selected by user.plus_subscription.
D15 no_interest_payment clears interest; minimum_payment clears only delinquency; MSI is inside both.

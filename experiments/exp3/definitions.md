DEFINITIONS
Vocabulary for the data schema, not rules. A definition alone is never a violation.
Absent numeric field = 0. If context has a field not described here, context wins.
D1 today — the only reference date.
D2 COMPLETE_MONTHS = monthly_summary months whose last day < today; CURRENT_PARTIAL = the month containing today.
D3 "last month" = calendar month before today's month. Not trailing 30 days, not today's month.
D4 net_purchases(M) = purchases − refunds.
D5 total_spending(M) = net_purchases(M) + transfers_to_others. transfers_own is never spending.
D6 avg_monthly_spending = mean total_spending over COMPLETE_MONTHS.
D7 monthly_summary = source for per-month aggregates. transactions = source for merchant, category and single-operation questions; it is a sample, not the full ledger.
D8 available(acc) = balance − on_hold.
D9 own_funds = Σ balance over deposit products, zero-balance included. credit_limit / available_credit are never own funds.
D10 net_position = own_funds − Σ current_debt.
D11 ACCOUNT_ROSTER = every element of accounts, zero-balance and credit ones included.
D12 liquid = checking + flexible savings − on_hold; Ahorro Fijo before maturity_date is not liquid.
D13 applicable_rate = account.annual_rate, else selected by user.plus_subscription.
D14 no_interest_payment clears interest; minimum_payment clears only delinquency; MSI is inside both.

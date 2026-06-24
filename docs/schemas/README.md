# Schemas

This directory deliberately ships **no local schema files**. The single source of truth for the trade-log row shape is upstream:

- [trade_log_v1.json](https://github.com/DaruFinance/quant-research-framework/blob/v2-main/docs/schemas/trade_log_v1.json) in `quant-research-framework`.

Why a pointer instead of a copy? A copy would drift the day a v1.1 column landed upstream and the audit suite started rejecting our output. Reference-by-pointer keeps both repos honest about the freeze.

Aux-log streams get their own schemas under this directory.

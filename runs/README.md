# Committed run artifacts

The CME depth-10 true-signed tapes are not redistributable (third-party provider, terms-bound).
In lieu of the raw feed, the per-run **result JSON/CSV** that the figure code renders directly are
committed here, so the paper's headline numbers are inspectable without the licensed data. Heavy
intermediates (per-fill `*_fills.parquet`, `*_windows.parquet`, OFI `feat/` panels) stay ignored;
given the licensed data the engines rebuild them from scratch.

| dir | paper section | carries |
|---|---|---|
| `maker_decomp/` | §5 (central) | decomposition, tick frontier, LVR-identity, by-root/by-regime |
| `maker_decomp_jun2022/` | §5 cross-regime | June-2022 stress-week decomposition + `regime_comparison.json` |
| `mm_full/` | §6, §8 | quoting null (`ASSEMBLY.json`), M4/M7/M8/M9 supporting measurements |
| `mm_full_jun2022/` | §5, §10(ii) | stress-week quoting null + harvestability (`methodology_retrofit.json`) |
| `xroot_ofi/` | §7 | cross-root integrated-OFI lead-lag frontier (`verdict_xroot_ofi.json`) |
| `xroot_ofi_jun2022/` | §10(ii) | stress-week OFI null |
| `h4_regime/` | §8 (H4) | regime-aware inventory-CVaR verdict |

Pre-registration files and hashes (`PHASE2_PREREGISTRATION.md`, `PRE-REGISTRATION-xroot-ofi.md`,
`PREREG_HASHES.txt`) are held in the study research archive, not this framework repo.

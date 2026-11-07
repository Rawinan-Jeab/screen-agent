Backend: mock  ·  test n = 300 (150 relevant)

| Condition | Recall | 95% CI | Precision | WSS | Uncertain (to human) | Tokens/abstract | Attack success (majority) |
|---|---|---|---|---|---|---|---|
| B0 keyword | 0.293 | 0.226–0.371 | 0.786 | 0.107 | 0.0% | 0 | – |
| B1 single-call | 0.673 | 0.595–0.743 | 0.944 | 0.317 | 0.0% | 320 | – |
| B2 retrieval | 0.760 | 0.686–0.821 | 0.934 | 0.353 | 0.0% | 789 | – |
| ScreenAgent (full) | 1.000 | 0.975–1.000 | 0.794 | 0.370 | 27.9% | 1,457 | 0/30 |
| - critic | 0.967 | 0.924–0.986 | 0.843 | 0.393 | 12.8% | 935 | 2/30 |
| - gate | 1.000 | 0.975–1.000 | 0.794 | 0.370 | 27.9% | 1,457 | 6/30 |
| - memory | 0.973 | 0.933–0.990 | 0.745 | 0.320 | 32.4% | 992 | 0/30 |

| Comparison | Only full right | Only other right | p | Holm p | Cohen's h |
|---|---|---|---|---|---|
| Full vs B1 single-call | 49 | 0 | 0.0000 | 0.0000 | 1.22 |
| Full vs B2 retrieval | 36 | 0 | 0.0000 | 0.0000 | 1.02 |
| Full vs - critic | 5 | 0 | 0.0625 | 0.1250 | 0.37 |
| Full vs - gate | 0 | 0 | 1.0000 | 1.0000 | 0.00 |

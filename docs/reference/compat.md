# Compatibility

Drop-in paths off other decision APIs. `mimir.compat.systemone.v1` converts Jev
`/v1/systemone` requests and answers, and `mimir.compat.laya.v1` offers
`load(...).predict(state, questions)` in Laya 0.3.20's shape, so existing callers move
by changing what they import.

::: mimir.compat.systemone.v1

::: mimir.compat.laya.v1

# Parent Review Summary

- Task: `b10624e5`
- Parent retest: `47/47` combos passed `50/50`
- Parent full fix: `index 17` recovered from codex output log, then generator overwrite/bit-allocation issues were fixed and retested to `50/50`
- Review status: all 47 combos treated as passed; `index 17` is `passed_with_parent_fix`

| index | combo_spec | test_result | final_status | notes |
|---:|---|---|---|---|
| 1 | D1/0+D10/0 | 50/50 (4.5ms/call) | passed | 50/50 parent retest passed |
| 2 | D1/4+D5/0 | 50/50 (2.4ms/call) | passed | 50/50 parent retest passed |
| 3 | D2/2+D3/0 | 50/50 (1.2ms/call) | passed | 50/50 parent retest passed |
| 4 | D4/1+D9/0 | 50/50 (1.7ms/call) | passed | 50/50 parent retest passed |
| 5 | D4/3+D11/0 | 50/50 (4.0ms/call) | passed | 50/50 parent retest passed |
| 6 | D5/0+D12/3 | 50/50 (3.6ms/call) | passed | 50/50 parent retest passed |
| 7 | D6/0+D15/1 | 50/50 (2.6ms/call) | passed | 50/50 parent retest passed |
| 8 | D1/0+D15/0 | 50/50 (2.6ms/call) | passed | 50/50 parent retest passed |
| 9 | D3/1+D13/0 | 50/50 (1.6ms/call) | passed | 50/50 parent retest passed |
| 10 | D6/0+D7/1 | 50/50 (1.7ms/call) | passed | 50/50 parent retest passed |
| 11 | D8/0+D16/0 | 50/50 (1.5ms/call) | passed | 50/50 parent retest passed |
| 12 | D2/1+D12/0 | 50/50 (1.8ms/call) | passed | 50/50 parent retest passed |
| 13 | D5/3+D8/3 | 50/50 (29.8ms/call) | passed | 50/50 parent retest passed |
| 14 | D4/0+D9/2 | 50/50 (3.3ms/call) | passed | 50/50 parent retest passed |
| 15 | D6/1+D10/1 | 50/50 (13.8ms/call) | passed | 50/50 parent retest passed |
| 16 | D2/0+D11/3 | 50/50 (4.0ms/call) | passed | 50/50 parent retest passed |
| 17 | D1/3+D8/0 | 50/50 (3.6ms/call) | passed_with_parent_fix | parent rescued missing code.py from codex output log and fixed generator overwrite/bit-allocation issues |
| 18 | D3/3+D4/1 | 50/50 (8.9ms/call) | passed | 50/50 parent retest passed |
| 19 | D5/1+D12/1 | 50/50 (3.3ms/call) | passed | 50/50 parent retest passed |
| 20 | D1/0+D14/0 | 50/50 (2.9ms/call) | passed | 50/50 parent retest passed |
| 21 | D3/2+D13/1 | 50/50 (3.0ms/call) | passed | 50/50 parent retest passed |
| 22 | D12/2+D16/3 | 50/50 (5.3ms/call) | passed | 50/50 parent retest passed |
| 23 | D2/3+D14/2 | 50/50 (3.5ms/call) | passed | 50/50 parent retest passed |
| 24 | D1/0+D5/0+D10/1 | 50/50 (3.0ms/call) | passed | 50/50 parent retest passed |
| 25 | D4/1+D9/0+D16/0 | 50/50 (1.6ms/call) | passed | 50/50 parent retest passed |
| 26 | D1/3+D8/0+D12/1 | 50/50 (3.4ms/call) | passed | 50/50 parent retest passed |
| 27 | D6/2+D11/0 | 50/50 (2.9ms/call) | passed | 50/50 parent retest passed |
| 28 | D7/2+D15/1 | 50/50 (5.2ms/call) | passed | 50/50 parent retest passed |
| 29 | D1/1+D9/1 | 50/50 (2.1ms/call) | passed | 50/50 parent retest passed |
| 30 | D2/2+D13/2 | 50/50 (1.0ms/call) | passed | 50/50 parent retest passed |
| 31 | D4/2+D11/1 | 50/50 (1.4ms/call) | passed | 50/50 parent retest passed |
| 32 | D8/0+D14/1 | 50/50 (4.5ms/call) | passed | 50/50 parent retest passed |
| 33 | D3/2+D16/1 | 50/50 (2.9ms/call) | passed | 50/50 parent retest passed |
| 34 | D4/2+D15/2 | 50/50 (1.9ms/call) | passed | 50/50 parent retest passed |
| 35 | D2/1+D6/0+D7/1 | 50/50 (1.4ms/call) | passed | 50/50 parent retest passed |
| 36 | D12/0+D16/2 | 50/50 (1.5ms/call) | passed | 50/50 parent retest passed |
| 37 | D9/3+D16/0 | 50/50 (1.6ms/call) | passed | 50/50 parent retest passed |
| 38 | D1/2+D9/3 | 50/50 (1.8ms/call) | passed | 50/50 parent retest passed |
| 39 | D5/0+D11/2 | 50/50 (2.4ms/call) | passed | 50/50 parent retest passed |
| 40 | D9/1+D13/3 | 50/50 (2.1ms/call) | passed | 50/50 parent retest passed |
| 41 | D8/1+D15/0 | 50/50 (2.6ms/call) | passed | 50/50 parent retest passed |
| 42 | D5/0+D10/0+D12/3 | 50/50 (2.6ms/call) | passed | 50/50 parent retest passed |
| 43 | D2/0+D5/2+D15/1 | 50/50 (2.1ms/call) | passed | 50/50 parent retest passed |
| 44 | D3/0+D4/1+D9/0 | 50/50 (1.6ms/call) | passed | 50/50 parent retest passed |
| 45 | D4/0+D6/3 | 50/50 (2.8ms/call) | passed | 50/50 parent retest passed |
| 46 | D8/2+D13/1 | 50/50 (1.5ms/call) | passed | 50/50 parent retest passed |
| 47 | D6/2+D11/0+D14/0 | 50/50 (14.7ms/call) | passed | 50/50 parent retest passed |

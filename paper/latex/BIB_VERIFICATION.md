# Bibliography verification (2026-11-16, ACL fact-check round)

All 100 entries of `references.bib` are cited in `main.tex` (100 of 100).
Each entry was checked three ways:

1. **Google Scholar**, searched by exact title in the user's browser through Kimi WebBridge (`scholar_run.py`, one query
   every 45-75 s). The column shows Scholar's author/venue line of the top hit. Scholar often lists the arXiv version
   first; for venue papers the venue was checked separately (arXiv comments, OpenReview, PMLR, proceedings).
2. **arXiv API** (`export.arxiv.org`): title and the full author list compared with the entry; entries without an id in
   the bib were found by title. Mismatches fixed on 11-16: OverFill (now COLM 2025), Tandem Transformers (ICML 2024),
   Afendulev et al. (Findings of EMNLP 2026), PLatour (now on arXiv), three titles printed in lower case.
3. **Claim check**: every sentence that cites an entry was checked against the cited paper (agy over all 80 cited
   abstracts, then two subagents over the full text of every paper whose description is specific). Corrected on 11-16:
   Ouro (last-loop KV reuse is an option it tests for decoding; it trains a fixed loop count, 8 then 4, with a loss at
   every loop), LoopDuet (prefill no deeper than decode; Huginn and recurrent OLMo-2), Tandem (not an encoder-decoder),
   BridgeKV (optional LoRA stage), MELT (sliding window of recent tokens), YOCO-U (only the global cache is
   loop-independent), Coconut (feeds the last hidden state as the next input embedding), LEV (reads a context, not a long
   one). FlashAttention (Dao et al., 2022) is now cited with FlashAttention-2 for the blockwise softmax.

| key | cited as | Google Scholar (top hit) | arXiv API | claim checked against |
|---|---|---|---|---|
| `cdb` | arXiv preprint arXiv:2608.09444, 2026 | K Schwethelm, D Rueckert, G Kaissis - arXiv preprint arXiv:2608.09444, | arXiv 2608.09444 | full text (App. A.1, Sec. 4.2) |
| `overfill` | Conference on Language Modeling (COLM), 2025 | **pending** (captcha) | arXiv 2508.08446 | full text (Sec. 2, 4.2); COLM 2025 on OpenReview |
| `fixedpoints2` | arXiv preprint arXiv:2610.06833, 2026 | B Huang, C Shi, J Chen, S Wen, Z Liu, E Xing… - arXiv preprint arXiv … | arXiv 2610.06833 | full text (Sec. 3.2) |
| `lla` | arXiv preprint arXiv:2607.15456, 2026 | JO Neill, F Reid - arXiv preprint arXiv:2607.15456, 2026 - arxiv.org | arXiv 2607.15456 | abstract |
| `benice` | arXiv preprint arXiv:2610.31633, 2026 | H Prairie, JM Ryan, T Berg-Kirkpatrick… - arXiv preprint arXiv …, 2026 | arXiv 2610.31633 | abstract |
| `elr` | arXiv preprint arXiv:2610.12582, 2026 | F Ladhak, S Acharya, SA Serano, S Sun… - arXiv preprint arXiv …, 2026  | arXiv 2610.12582 | full text (Sec. 5.3) |
| `plt` | arXiv preprint arXiv:2510.24824, 2025 | **pending** (captcha) | arXiv 2510.24824 | abstract |
| `loopcoder2` | arXiv preprint arXiv:2606.18023, 2026 | J Yang, S Guo, W Zhang, T Zheng, Y Du, HS Li… - arXiv preprint arXiv … | arXiv 2606.18023 | full text (Sec. 3.1) |
| `sharedmem` | arXiv preprint arXiv:2610.02383, 2026 | **pending** (captcha) | arXiv 2610.02383 | abstract |
| `decodebranch` | arXiv preprint arXiv:2608.12385, 2026 | L Liu, M Wang, T Zhao - arXiv preprint arXiv:2608.12385, 2026 - arxiv. | arXiv 2608.12385 | abstract |
| `phd` | arXiv preprint arXiv:2504.14992, 2025 | **pending** (captcha) | arXiv 2504.14992 | full text (Sec. 2.2) |
| `sps` | arXiv preprint arXiv:2609.03807, 2026 | **pending** (captcha) | arXiv 2609.03807 | full text (Sec. 6.4) |
| `heterofold` | arXiv preprint arXiv:2609.32259, 2026 | VD Yun, W Lim, H Yoo, S Yoo, SP Karimireddy… - arXiv preprint arXiv …, | arXiv 2609.32259 | abstract |
| `swat` | arXiv preprint arXiv:2610.15095, 2026 | **pending** (captcha) | arXiv 2610.15095 | abstract |
| `cachelink` | arXiv preprint arXiv:2610.15509, 2026 | H Qiu, Y Tang, Z Song, R Han, Z Qiao… - arXiv preprint arXiv …, 2026 - | arXiv 2610.15509 | full text (Sec. 5) |
| `c2c` | International Conference on Learning Representations (ICLR), 2026 | T Fu, Z Min, H Zhang, J Yan, G Dai… - International …, 2026 - proceedi | arXiv 2510.03215 | abstract |
| `bellows` | arXiv preprint arXiv:2610.35023, 2026 | J Zhang, H Chen - arXiv preprint arXiv:2610.35023, 2026 - arxiv.org | arXiv 2610.35023 | abstract |
| `accordion` | arXiv preprint arXiv:2610.32492, 2026 | Y Zhang, W Ding, K Zheng, Y Ding, Z Zhang - arXiv preprint arXiv …, 20 | arXiv 2610.32492 | standard use (benchmark, method or tool) |
| `padd` | Findings of the Association for Computational Linguistics: A, 2025 | **pending** (captcha) | arXiv 2503.08524 | abstract |
| `pop` | Findings of the Association for Computational Linguistics: A, 2026 | **pending** (captcha) | arXiv 2602.03295 | abstract |
| `speed` | arXiv preprint arXiv:2605.06105, 2026 | **pending** (captcha) | arXiv 2605.06105 | abstract |
| `ouro` | arXiv preprint arXiv:2510.25741, 2025 | **pending** (captcha) | arXiv 2510.25741 | full text (KV reuse is a decode option, Sec. 5.4.2; training loop count, Table 1) |
| `huginn` | Advances in Neural Information Processing Systems (NeurIPS), 2025 | J Geiping, S McLeish, N Jain… - Advances in …, 2026 - proceedings.neur | arXiv 2502.05171 | full text (Sec. 3.3) |
| `ruler` | Conference on Language Modeling (COLM), 2024 | **pending** (captcha) | arXiv 2404.06654 | standard use (benchmark, method or tool) |
| `helmet` | International Conference on Learning Representations (ICLR), 2025 | H Yen, T Gao, M Hou, K Ding… - International …, 2025 - proceedings.icl | arXiv 2410.02694 | standard use (benchmark, method or tool) |
| `lora` | International Conference on Learning Representations (ICLR), 2022 | **pending** (captcha) | arXiv 2106.09685 | standard use (benchmark, method or tool) |
| `mmlu` | International Conference on Learning Representations (ICLR), 2021 | **pending** (captcha) | arXiv 2009.03300 | standard use (benchmark, method or tool) |
| `gsm8k` | arXiv preprint arXiv:2110.14168, 2021 | **pending** (captcha) | arXiv 2110.14168 | standard use (benchmark, method or tool) |
| `bbh` | Findings of the Association for Computational Linguistics: A, 2023 | **pending** (captcha) | arXiv 2210.09261 | standard use (benchmark, method or tool) |
| `mmlupro` | Advances in Neural Information Processing Systems (NeurIPS),, 2024 | **pending** (captcha) | arXiv 2406.01574 | standard use (benchmark, method or tool) |
| `arc` | arXiv preprint arXiv:1803.05457, 2018 | **pending** (captcha) | arXiv 1803.05457 | standard use (benchmark, method or tool) |
| `math` | Advances in Neural Information Processing Systems (NeurIPS),, 2021 | **pending** (captcha) | arXiv 2103.03874 | standard use (benchmark, method or tool) |
| `humaneval` | arXiv preprint arXiv:2107.03374, 2021 | **pending** (captcha) | arXiv 2107.03374 | standard use (benchmark, method or tool) |
| `rope` | Neurocomputing, 2024 | **pending** (captcha) | arXiv 2104.09864 | standard use (benchmark, method or tool) |
| `mor` | Advances in Neural Information Processing Systems (NeurIPS), 2025 | **pending** (captcha) | arXiv 2507.10524 | abstract |
| `ut` | International Conference on Learning Representations (ICLR), 2019 | **pending** (captcha) | arXiv 1807.03819 | standard use (benchmark, method or tool) |
| `evalplus` | Advances in Neural Information Processing Systems (NeurIPS), 2023 | **pending** (captcha) | arXiv 2305.01210 | standard use (benchmark, method or tool) |
| `flashattn2` | International Conference on Learning Representations (ICLR), 2024 | T Dao - International Conference on Learning Representations, 2024 - p | arXiv 2307.08691 | standard use (benchmark, method or tool) |
| `lstmloop` | arXiv preprint arXiv:2610.33360, 2026 | **pending** (captcha) | arXiv 2610.33360 | abstract |
| `lightman2023lets` | International Conference on Learning Representations (ICLR), 2024 | **pending** (captcha) | arXiv 2305.20050 | standard use (benchmark, method or tool) |
| `yoco` | Advances in Neural Information Processing Systems (NeurIPS), 2024 | **pending** (captcha) | arXiv 2405.05254 | standard use (benchmark, method or tool) |
| `yocou` | arXiv preprint arXiv:2604.01220, 2026 | **pending** (captcha) | arXiv 2604.01220 | full text (Sec. 3.1, Fig. 1) |
| `kite` | arXiv preprint arXiv:2609.27294, 2026 | Z Hu, Y Wei, J Zhou, Y Zhou, J Li, X Chen, Y Li… - arXiv preprint arXi | arXiv 2609.27294 | full text (Sec. 1, 3.2) |
| `lev` | arXiv preprint arXiv:2610.20951, 2026 | D Song, Z Yan, Y Zhang, L Sun - arXiv preprint arXiv:2610.20951, 2026  | arXiv 2610.20951 | full text (Sec. 4.1, 5) and the LevBench / Lev-train-v2 dataset cards |
| `gdn` | International Conference on Learning Representations (ICLR), 2025 | S Yang, J Kautz, A Hatamizadeh - International Conference on …, 2025 - | arXiv 2412.06464 | standard use (benchmark, method or tool) |
| `hymba` | International Conference on Learning Representations (ICLR), 2025 | X Dong, Y Fu, S Diao, W Byeon, Z Chen… - International …, 2025 - proce | arXiv 2411.13676 | standard use (benchmark, method or tool) |
| `afendulev` | Findings of the Association for Computational Linguistics: E, 2026 | K Afendulev, A Dontsov, E Tutubalina… - Findings of the …, 2026 - acla | arXiv 2609.04434 | abstract; Findings of EMNLP 2026 (arXiv comment) |
| `latentport` | arXiv preprint arXiv:2609.25053, 2026 | SP Villani - arXiv preprint arXiv:2609.25053, 2026 - arxiv.org | arXiv 2609.25053 | abstract |
| `melt` | arXiv preprint arXiv:2605.07721, 2026 | **pending** (captcha) | arXiv 2605.07721 | full text (Sec. 3.2, 4.1) |
| `hellaswag` | Proceedings of the 57th Annual Meeting of the Association fo, 2019 | **pending** (captcha) | no arXiv version | standard use (benchmark, method or tool) |
| `winogrande` | Proceedings of the AAAI Conference on Artificial Intelligenc, 2020 | **pending** (captcha) | no arXiv version | standard use (benchmark, method or tool) |
| `flashloop` | arXiv preprint arXiv:2609.29812, 2026 | W Yang, S Liu - arXiv preprint arXiv:2609.29812, 2026 - arxiv.org | arXiv 2609.29812 | abstract |
| `giannou2023looped` | Proceedings of the 40th International Conference on Machine , 2023 | A Giannou, S Rajput, J Sohn, K Lee… - International …, 2023 - proceedi | arXiv 2301.13196 | standard use (benchmark, method or tool) |
| `saunshi2025latent` | International Conference on Learning Representations (ICLR), 2025 | **pending** (captcha) | arXiv 2502.17416 | abstract |
| `relaxedrecursive` | International Conference on Learning Representations (ICLR), 2025 | **pending** (captcha) | arXiv 2410.20672 | full text (Sec. 2.3) |
| `loopedlengthgen` | International Conference on Learning Representations (ICLR), 2025 | **pending** (captcha) | arXiv 2409.15647 | abstract |
| `albert` | International Conference on Learning Representations (ICLR), 2020 | Z Lan, M Chen, S Goodman, K Gimpel… - arXiv preprint arXiv …, 2019 - a | arXiv 1909.11942 | standard use (benchmark, method or tool) |
| `act` | arXiv preprint arXiv:1603.08983, 2016 | A Graves - arXiv preprint arXiv:1603.08983, 2016 - arxiv.org | arXiv 1603.08983 | standard use (benchmark, method or tool) |
| `pondernet` | arXiv preprint arXiv:2107.05407, 2021 | **pending** (captcha) | arXiv 2107.05407 | standard use (benchmark, method or tool) |
| `calm` | Advances in Neural Information Processing Systems (NeurIPS), 2022 | T Schuster, A Fisch, J Gupta… - Advances in …, 2022 - proceedings.neur | arXiv 2207.07061 | standard use (benchmark, method or tool) |
| `layerskip` | Proceedings of the 62nd Annual Meeting of the Association fo, 2024 | M Elhoushi, A Shrivastava, D Liskovich… - Proceedings of the …, 2024 - | arXiv 2404.16710 | standard use (benchmark, method or tool) |
| `mod` | arXiv preprint arXiv:2404.02258, 2024 | **pending** (captcha) | arXiv 2404.02258 | standard use (benchmark, method or tool) |
| `deebert` | Proceedings of the 58th Annual Meeting of the Association fo, 2020 | J Xin, R Tang, J Lee, Y Yu, J Lin - … of the 58th annual meeting of …, | arXiv 2004.12993 | standard use (benchmark, method or tool) |
| `depthadaptive` | International Conference on Learning Representations (ICLR), 2020 | M Elbayad, J Gu, E Grave, M Auli - arXiv preprint arXiv:1910.10073, 20 | arXiv 1910.10073 | standard use (benchmark, method or tool) |
| `specdecoding` | Proceedings of the 40th International Conference on Machine , 2023 | **pending** (captcha) | arXiv 2211.17192 | standard use (benchmark, method or tool) |
| `specsampling` | arXiv preprint arXiv:2302.01318, 2023 | **pending** (captcha) | arXiv 2302.01318 | standard use (benchmark, method or tool) |
| `mqa` | arXiv preprint arXiv:1911.02150, 2019 | **pending** (captcha) | arXiv 1911.02150 | standard use (benchmark, method or tool) |
| `gqa` | Proceedings of the 2023 Conference on Empirical Methods in N, 2023 | J Ainslie, J Lee-Thorp, M De Jong… - Proceedings of the …, 2023 - acla | arXiv 2305.13245 | standard use (benchmark, method or tool) |
| `cla` | Advances in Neural Information Processing Systems (NeurIPS), 2024 | W Brandon, M Mishra, A Nrusimha… - Advances in …, 2024 - proceedings.n | arXiv 2405.12981 | standard use (benchmark, method or tool) |
| `mlkv` | Findings of the Association for Computational Linguistics: N, 2025 | **pending** (captcha) | arXiv 2406.09297 | standard use (benchmark, method or tool) |
| `streamingllm` | International Conference on Learning Representations (ICLR), 2024 | **pending** (captcha) | arXiv 2309.17453 | standard use (benchmark, method or tool) |
| `h2o` | Advances in Neural Information Processing Systems (NeurIPS), 2023 | Z Zhang, Y Sheng, T Zhou, T Chen… - Advances in neural …, 2023 - proce | no arXiv version | standard use (benchmark, method or tool) |
| `snapkv` | Advances in Neural Information Processing Systems (NeurIPS), 2024 | **pending** (captcha) | arXiv 2404.14469 | standard use (benchmark, method or tool) |
| `pyramidkv` | arXiv preprint arXiv:2406.02069, 2024 | **pending** (captcha) | arXiv 2406.02069 | standard use (benchmark, method or tool) |
| `vllm` | Proceedings of the 29th Symposium on Operating Systems Princ, 2023 | **pending** (captcha) | arXiv 2309.06180 | standard use (benchmark, method or tool) |
| `distserve` | 18th USENIX Symposium on Operating Systems Design and Implem, 2024 | Y Zhong, S Liu, J Chen, J Hu, Y Zhu, X Liu… - … USENIX symposium on …, | arXiv 2401.09670 | standard use (benchmark, method or tool) |
| `splitwise` | 51st ACM/IEEE Annual International Symposium on Computer Arc, 2024 | **pending** (captcha) | arXiv 2311.18677 | standard use (benchmark, method or tool) |
| `sarathiserve` | 18th USENIX Symposium on Operating Systems Design and Implem, 2024 | **pending** (captcha) | arXiv 2403.02310 | standard use (benchmark, method or tool) |
| `mooncake` | 23rd USENIX Conference on File and Storage Technologies (FAS, 2025 | **pending** (captcha) | arXiv 2407.00079 | standard use (benchmark, method or tool) |
| `kd` | arXiv preprint arXiv:1503.02531, 2015 | G Hinton, O Vinyals, J Dean - arXiv preprint arXiv:1503.02531, 2015 -  | arXiv 1503.02531 | standard use (benchmark, method or tool) |
| `coconut` | Conference on Language Modeling (COLM), 2025 | S Hao, S Sukhbaatar, DJ Su, X Li, Z Hu… - arXiv preprint arXiv …, 2024 | arXiv 2412.06769 | full text |
| `pausetokens` | International Conference on Learning Representations (ICLR), 2024 | **pending** (captcha) | arXiv 2310.02226 | standard use (benchmark, method or tool) |
| `dory` | arXiv preprint arXiv:2611.03998, 2026 | M Fishman, S Sun, S Acharya, F Ladhak… - arXiv preprint arXiv …, 2026  | arXiv 2611.03998 | full text (abstract, Sec. 6.1) |
| `seedemit` | arXiv preprint arXiv:2611.06084, 2026 | **pending** (captcha) | arXiv 2611.06084 | abstract |
| `platour` | arXiv preprint arXiv:2611.06976, 2026 | **pending** (captcha) | arXiv 2611.06976 | full text (Sec. 4) |
| `swiftkv` | Proceedings of the 2025 Conference on Empirical Methods in N, 2025 | **pending** (captcha) | arXiv 2410.03960 | abstract |
| `skipdecode` | arXiv preprint arXiv:2307.02628, 2023 | **pending** (captcha) | arXiv 2307.02628 | standard use (benchmark, method or tool) |
| `llm2slm` | arXiv preprint arXiv:2402.16844, 2024 | B Bergner, A Skliar, A Royer, T Blankevoort… - arXiv preprint arXiv …, | arXiv 2402.16844 | full text (Sec. 4.1) |
| `tandem` | Proceedings of the 41st International Conference on Machine , 2024 | **pending** (captcha) | arXiv 2402.08644 | full text (Sec. 1); ICML 2024, PMLR 235:42906-42917 |
| `deepshallow` | International Conference on Learning Representations (ICLR), 2021 | J Kasai, N Pappas, H Peng, J Cross… - arXiv preprint arXiv …, 2020 - a | arXiv 2006.10369 | standard use (benchmark, method or tool) |
| `loopduet` | arXiv preprint arXiv:2611.07211, 2026 | **pending** (captcha) | arXiv 2611.07211 | full text (abstract, Sec. 1, 4.1) |
| `bridgekv` | arXiv preprint arXiv:2611.11761, 2026 | P Behnam, D Sanyal, P Ponnusamy, KM Ravi… - arXiv preprint arXiv …, 20 | arXiv 2611.11761 | full text (abstract, App. C) |
| `flashattn` | Advances in Neural Information Processing Systems (NeurIPS), 2022 | **pending** (captcha) | arXiv 2205.14135 | full text (Sec. 3.1, tiling) |
| `mbpp` | arXiv preprint arXiv:2108.07732, 2021 | **pending** (captcha) | arXiv 2108.07732 | standard use (benchmark, method or tool) |
| `minerva` | Advances in Neural Information Processing Systems (NeurIPS), 2022 | **pending** (captcha) | arXiv 2206.14858 | standard use (benchmark, method or tool) |
| `lmeval` | arXiv preprint arXiv:2405.14782, 2024 | **pending** (captcha) | arXiv 2405.14782 | standard use (benchmark, method or tool) |
| `openthoughts` | arXiv preprint arXiv:2506.04178, 2025 | **pending** (captcha) | arXiv 2506.04178 | standard use (benchmark, method or tool) |
| `tulu3` | arXiv preprint arXiv:2411.15124, 2024 | **pending** (captcha) | arXiv 2411.15124 | standard use (benchmark, method or tool) |
| `longlora` | International Conference on Learning Representations (ICLR), 2024 | **pending** (captcha) | arXiv 2309.12307 | standard use (benchmark, method or tool) |
| `fineweb` | Advances in Neural Information Processing Systems (NeurIPS),, 2024 | **pending** (captcha) | arXiv 2406.17557 | standard use (benchmark, method or tool) |

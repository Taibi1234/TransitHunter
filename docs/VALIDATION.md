\# TransitHunter validation record



\## 1. Validation claim



TransitHunter v1.0.0 demonstrates blind recovery of a known periodic transit

signal from a real Kepler light curve, followed by limited automated screening

and catalogue identification.



This is a known-planet re-detection benchmark. It is not a new-planet discovery,

an independent planetary confirmation, or a mission-wide performance study.



\## 2. Data provenance



\- Target: KIC 11904151 (Kepler-10)

\- Mission: Kepler

\- Observation interval: Quarter 3

\- Product selection: Kepler author, long cadence

\- Archive: Mikulski Archive for Space Telescopes (MAST)

\- Access library: Lightkurve

\- Downloaded rows: 4,228

\- Acquisition quality bitmask: 0, preserving all downloaded cadences

\- Raw export: time, flux, flux uncertainty, and original quality flags



The raw and processed datasets are generated locally and are not committed to

the repository.



\## 3. Preprocessing



Operations were applied in the following order:



1\. Retain measurements with quality flag zero.

2\. Remove non-finite time, flux, and uncertainty values.

3\. Require positive finite flux uncertainties.

4\. Sort by timestamp.

5\. Divide flux and flux uncertainty by median flux.

6\. Remove extreme positive outliers only.



Results:



| Stage | Rows |

|---|---:|

| Input | 4,228 |

| Non-zero quality flags removed | 594 |

| Non-finite rows removed | 1 |

| Non-positive uncertainties removed | 0 |

| Positive outliers removed | 0 |

| Output | 3,633 |



No negative-outlier clipping, smoothing, interpolation, or detrending was

applied during this stage.



\## 4. Detrending



\- Method: gap-aware rolling median

\- Window: 1.0 day

\- Gap threshold: 0.5 day

\- Negative-deviation protection threshold: 3 robust sigma

\- Protection strategy: one fixed mask, followed by final baseline estimation

\- Observing segments: 4

\- Protected negative deviations: 142

\- Rows removed: 0



Protected measurements were excluded only from baseline fitting. They remained

available to the BLS search. No published ephemeris was used.



\## 5. Blind BLS search



\- Algorithm: Astropy BoxLeastSquares

\- Objective: likelihood

\- Period interval: 0.5–10.0 days

\- Trial durations: 0.04, 0.06, 0.08, and 0.12 days

\- Frequency factor: 1.0

\- Minimum required transits: 3

\- Period-grid evaluations: 378,377

\- Published period supplied: no

\- Published epoch supplied: no



\## 6. Recovery result



| Quantity | Blind result | Catalogue comparison |

|---|---:|---:|

| Period | 0.8374290408 d | 0.837491225 d |

| Duration | 1.92 h | 1.796 h |

| Depth | 151.33 ppm | 190.4 ppm |

| Object | Not supplied | Kepler-10 b / K00072.01 |



The recovered and catalogue periods differ by approximately 5.37 seconds.

Duration and depth estimates are method-dependent and are not expected to be

identical to catalogue model fits.



!\[Blind BLS diagnostic](images/kepler10-bls.png)



\## 7. Automated screening



\- Measurable primary events: 98

\- Odd transits: 48

\- Even transits: 50

\- Odd depth: 145.68 ppm

\- Even depth: 155.08 ppm

\- Odd/even difference: -9.40 ppm

\- Formal odd/even significance: -1.878 sigma

\- Phase-0.5 secondary depth: 7.69 ppm

\- Formal secondary SNR: 3.126

\- Screening flag: significant\_secondary\_eclipse

\- Screening status: flagged\_for\_false\_positive\_review



The secondary measurement is marginal and relies on formal local uncertainties.

It is not sufficient to classify the target as an eclipsing binary or to confirm

a planetary occultation. Correlated noise and residual systematics require

further investigation.



\## 8. Catalogue identification



The exact KIC identifier was used to query the cumulative KOI table. The host

name derived from that response was used to query the confirmed-planet table.



Results:



\- Confirmed catalogue period match: Kepler-10 b

\- KOI period match: K00072.01

\- KOI disposition: CONFIRMED

\- Final catalogue status: known\_confirmed\_planet\_signal



This catalogue step prevents the blindly recovered signal from being presented

as a new candidate.



\## 9. Injection/recovery experiment



The strongest known signal was masked using its blindly recovered period,

epoch, and duration. Nine box-shaped signals were then injected into the

remaining detrended measurements. Only the strongest BLS peak from each trial

was evaluated.



| Injected depth | Recovered |

|---:|---:|

| 100 ppm | 2 / 3 |

| 250 ppm | 3 / 3 |

| 500 ppm | 3 / 3 |

| Total | 8 / 9 |



Recovery required fundamental-period agreement, epoch agreement, formal depth

SNR of at least 5, and periodogram power above the zero-depth control. Period

aliases were not counted as recoveries.



The zero-depth control produced a 141 ppm residual signal at 9.07 days with

formal depth SNR 25. The missed 100 ppm injection at 4.1 days did not exceed

this control. This result demonstrates the importance of controls and vetting.



!\[Injection/recovery diagnostic](images/injection-recovery.png)



This nine-trial experiment is a bounded software benchmark. It does not

estimate general detection completeness or discovery performance.



\## 10. Software verification



\- Automated tests: 20

\- Network access required by tests: no

\- Initial GitHub Actions CI: passed

\- End-to-end cached pipeline reproduction: passed



The tests cover acquisition exports, masked arrays, deterministic caching,

transit preservation, gap handling, blind synthetic BLS recovery, phase

folding, odd/even screening, secondary screening, catalogue parsing, CLI

argument wiring, injection scoring, and pipeline orchestration.



\## 11. Reproduction



From the repository root in Windows PowerShell:



```powershell

py -3.11 -m venv .venv

.\\.venv\\Scripts\\python.exe -m pip install -e ".\[dev]"

.\\.venv\\Scripts\\python.exe -m pytest -q

.\\.venv\\Scripts\\python.exe -m transithunter.pipeline "KIC 11904151" --mission Kepler --author Kepler --cadence long --quarter 3


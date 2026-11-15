# Streuende Lieferzeit – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-streuende-lieferzeit-demo.streamlit.app/)** -->

Demo der Linie **Rückkopplung und Verzögerung** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie fragt, wann ein Kreis aus Bestand, Fluss und Verzögerung schwingt, wächst oder kippt und was davon an der Regel, an der Verzögerung oder am Rechnen hängt. Diese Demo ändert nur **die Form der Lieferzeit** bei gleichem Mittel.
Sie hebt eine Schwäche von [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) auf: Dort gilt die Grenze α_krit = 2·sin(π/(4L − 2)) für eine **feste** Lieferzeit L, aber echte Lieferzeiten streuen. Zugleich ist sie ein Kontrast zu [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo) (feste stetige Verzögerung, a·τ = π/2): gleiche mittlere Verzögerung, andere Form. Nachbarn der Linie: [`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo) (Leitungsgewicht β), [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (Meldeverzug), [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo) (Frequenzgang); mit Klemmung und Chaos beschäftigt sich [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo).

**Das Wort „Erlang“** meint hier die Verteilung der Lieferzeit (Summe aus k gleichen geometrischen Zeiten), nicht die Warteschlangenformel der Erlang-Demos des Portfolios.

## Warum dieses Problem

Ein Lager bestellt nach der linearen Regel u = −α·(Bestand − Soll) nach; kommt die Lieferung nach genau τ Perioden, schwingt der Bestand ab α_krit = 2·sin(π/(4τ − 2)) auf (L = τ). Die Demo fragt: **Streut die Lieferzeit bei gleichem Mittel, was gilt dann noch von dieser Grenze, und was ändert sich an Schwankung, Abklingzeit und Einzelbestellungen?**
Die Antwort ist nicht einheitlich: Die Grenze steigt (bei β = 0, in den gerechneten Formen), die Schwankung bei weißem Rauschen sinkt (im anteiligen Modell), aber das Abklingen kann sehr lange dauern, und kommt jede Bestellung als Ganzes nach einer zufälligen Lieferzeit, liegt die Grenze niedriger als im anteiligen Modell (in den gerechneten Fällen aber noch über der festen Lieferzeit) und der Bestand schwankt auch bei konstantem Bedarf.
Alles gilt für dieses Modell (ein einzelnes Lager, linear, ohne Klemmung, kein Meldeverzug); eine Aussage über reale Lieferzeitverteilungen ist damit nicht gemacht.

## Modell

Abweichungen vom Gleichgewicht, Zeit in Perioden: Bestand i (minus Soll), Bestellung u (minus Mittelbedarf), Bedarfsabweichung e.

- Regel: **u_t = −α·(i_t + β·w_t)**, w_t die Leitung (was bestellt und noch nicht angekommen ist); **β = 0** rechnet sie nicht mit, **β = 1** voll (Bestandsposition).
- Bestand: **i_{t+1} = i_t + Σ_l f_l·u_{t+1−l} − e_t**; f_l ist der Anteil einer Bestellung, der nach l Perioden ankommt (**anteilig**, Erwartungswert-Modell).
- Hauptfamilie der Lieferzeit-Verteilung (eigene Wahl): Summe aus **k** geometrischen Zeiten mit Fortschrittswahrscheinlichkeit **p** je Periode, mindestens k Perioden, **Mittel τ = k/p**, f_l = C(l−1, k−1)·p^k·q^(l−k). **p = 1** ist die feste Lieferzeit L = k. In der App ist τ ein Regler, p = k/τ ergibt sich; k > τ wird auf τ begrenzt (fest).
- **Einzelbestellung:** Jede Bestellung O = D + u kommt als Ganzes nach einer gezogenen Lieferzeit, Bestellungen dürfen sich überholen; der Mittelwert folgt dem anteiligen Modell, die Streuung nicht.
- **Charakteristisches Polynom** (eigene Herleitung in diesem Modell): β = 0: (z − 1)(z − q)^k + α·p^k·z; β = 1: (z − (1 − α))(z − q)^k. Die Grenze ist der kleinste positive reelle Wert von α(z) = (1 − z)(z − q)^k/(p^k·z) auf dem Einheitskreis (Wurzelortskurve); k = 1: α_krit = 4τ − 2 (Wurzel z = −1, Periode 2), k = 2: α_krit = p(2 − p).
- Die Formeln stehen im Expander 📐 der App; α, β, τ, k und der Zufalls-Seed sind Regler der Seitenleiste, die Regler einzelner Experimente stehen bei dem Experiment.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Eigenwerte der Zustandsmatrix** (`slz_dynamics.chain_matrix`, `spectral_radius`) | Zustand (i, x_1, …, x_k), x_j Inhalt der Stufe j; Spektralradius unter 1 heißt stabil | Eigenwerte langer Matrizen mit vielen Wurzeln nahe 0 sind schlecht konditioniert: gerechnet werden nur Ketten bis k = 24; Verteilungsmatrizen (bis 28 Zustände) nur für die Zählung der Wiederkehr und in Tests |
| **Wurzelortskurve** (`critical_alpha_chain`, `critical_alpha_dist`) | Reelle positive Werte von α(z) auf dem Einheitskreis (Vorzeichenwechsel des Imaginärteils auf einem Winkelraster, danach Bisektion); die kleinste ist die Grenze | findet nur Kreuzungen, die das Raster als Vorzeichenwechsel sieht (Berührungen ohne Wechsel nicht); gilt für β = 0, bei β = 1 ist α_krit = 2 hergeleitet |
| **Lyapunov-Gleichung** (`variances`) | var(i) und var(u) bei weißem Rauschen aus S = A·S·Aᵀ + b·bᵀ | nur stabile Fälle; Normalverteilung nicht nötig, nur Varianz |
| **Impulsantwort, Abklingzeit** (`impulse_response`, `decay_time`) | Bedarfsstoß 1; letzte Periode, in der der Betrag des Bestands über 1 % seiner Spitze liegt | Obergrenze 100 000 Perioden; „1 %“ ist eine Wahl |
| **Zweites Moment der Einzelbestellung** (`ms_operator`, `ms_stationary`, `ms_critical`) | Lineares System für E[z zᵀ] und die Summe der Quadrate der Bestellklumpen je Stufe; stationär genau bei Spektralradius unter 1 | Größe (k+1)² + k: in der App nur für k ≤ 8; Grenze auf einem Raster mit 20 % Schritt gesucht, schmalere Instabilitätsfenster sind nicht ausgeschlossen |
| **Simulation** (`simulate_individual`, `simulate_fluid`) | Einzelbestellung: je Bestellung gezogene Lieferzeit (Summe aus k geometrischen Zeiten, auf 30·τ, höchstens 720 gekappt); anteilig: Kette mit Normalrauschen | Stichprobenfehler; Bänder in den Tests |

Alle Zufallsfolgen kommen aus einem eigenen **SplitMix64** (`slz_rng.py`, vektorisiert gegen die Klasse und gegen den Referenzwert für den Zustand 0 geprüft); ein Zufalls-Seed je Zweck und je Verteilung wird über `child_seed` abgeleitet, nie über Python-`hash()`.
Die Live-Rechnung der App braucht für alle Abschnitte zusammen je Eingabe etwa 0,4 bis 1,7 Sekunden (Messung auf dem Entwicklungsrechner, nicht festgeschrieben; der Abschnitt Einzelbestellung ist mit bis zu 1,3 Sekunden bei 600 Läufen, τ = 24 und k = 12 der teuerste); die Reihe der 1 000 Zufallsverteilungen ist **vorgerechnet** (`data/slz_frozen.json`, erzeugt von `tools/make_frozen.py`, etwa 30 Sekunden).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Perioden, linear, ohne Klemmung, β = 0 und anteilig, wo nichts anderes steht; „fest“ heißt k = τ. Die Tests prüfen Bänder (Rechenunterschiede der Plattformen, Stichproben); die Spalte „Test“ nennt die Funktion in `tests/test_claims.py`.
Die Verteilungsfamilie ist eigene Wahl (Summe geometrischer Zeiten); die Befunde gelten für diese Familie und für die genannten Formen, nicht für beliebige Lieferzeiten.

| Frage | Befund | Test |
|---|---|---|
| Wie hoch liegt die Grenze bei gleichem Mittel? | **τ = 12:** fest 0,1365, k = 6: 0,1541, k = 4: 0,1767, k = 3: 0,2065, k = 2: 0,3056 (das 2,24fache), k = 1: 46 (= 4τ − 2). **τ = 60:** fest 0,0264, k = 10: 0,0295, k = 3: 0,0438, k = 2: 0,0656 (das 2,48fache), k = 1: 238. Geschlossen (eigene Herleitung, Jury-Kriterium): k = 1: 4τ − 2, k = 2: 4(τ − 1)/τ²; beide in je 11 Fällen (τ von 2 bis 1200) gleich der Messung | `test_limits_at_tau_12_and_60`, `test_limit_formulas_k1_and_k2_in_eleven_cases` |
| Nähert sich das dem stetigen Grenzfall? | α_krit·τ für k = 2, 3, 4, 6, 10: bei τ = 12 3,667 / 2,478 / 2,121 / 1,850 / 1,676, bei τ = 1200 3,9967 / 2,6648 / 2,2727 / 1,9782 / 1,7916, stetig 4,0000 / 2,6667 / 2,2742 / 1,9794 / 1,7927; feste Lieferzeit 1,638 (τ = 12), 1,584 (τ = 60), stetig π/2 = 1,5708 | `test_limit_times_tau_approaches_the_continuous_values` |
| Wie viel höher liegt die Grenze der Verzögerung dritter Ordnung (k = 3) als die der festen Lieferzeit? | Verhältnis 1,335 / 1,513 / 1,604 / 1,660 / 1,679 bei τ = 6 / 12 / 24 / 60 / 120, stetig 1,698 (= (8/3)/(π/2), exakt aus dem Hurwitz-Kriterium): **+51 % bei τ = 12, +66 % bei τ = 60, +70 % nur im stetigen Grenzfall.** DELAY3 in Vensim ist eine Verzögerung dritter Ordnung (Dokumentation von der unabhängigen Prüfung gelesen); ob sie die „Standardverzögerung“ der Systemdynamik-Software ist, ist nicht belegt. Die Dokumentation vergleicht keine Grenzen, die Zahlen sind diese Messung | `test_third_order_against_fixed_ratios_and_percentages`, `test_hand_continuous_limits_by_hurwitz` (in `test_dynamics.py`) |
| Gilt „Streuung hebt die Grenze“ für jede Form? (Mittel 4) | Fest 0,4450; zwei Punkte 3 und 5: 0,4940, 2 und 6: 0,7138, 1 und 7: 2,0000; gleichverteilt 3 bis 5: 0,4765, 2 bis 6: 0,5496, 1 bis 7: 0,6932. **1 000 Zufallsverteilungen** (Zufalls-Seed 20261115, Mittel 4, auf 1 bis 5 bis 1 bis 8, eigene Familie): die Grenze liegt in **999 über der festen**, in 1 innerhalb 10⁻⁶ der festen (eine fast feste Verteilung), in 0 darunter; Median das 1,33fache, größtes das 9,15fache; Median je Form 1,39 (glatt), 1,38 (gemischt), 1,22 (spitz). **Ausnahme zur Form „stabil unterhalb der ersten Grenze“:** in 4 der 1 000 kehrt die Stabilität oberhalb der ersten Grenze auf dem α-Raster (Schritt 0,01 bis 2,6) zurück (nicht näher untersucht). Gegenprobe mit anderem Mittel (je 1 000 vorgerechnet, die ersten 250 neu gerechnet): keine darunter bei Mittel 3, 5, 6; Median 1,53 / 1,22 / 1,17; keine Rückkehr | `test_other_shapes_at_mean_4`, `test_random_distributions_mean_4_counts_and_the_exception`, `test_random_distributions_mean_4_recomputed_equal_the_frozen_row`, `test_the_returning_distributions_have_two_crossings_and_stable_raster_points_above_the_first_limit`, `test_random_distributions_other_means_partial_recomputation` |
| Wird die Schwankung kleiner? (weißes Rauschen, anteilig) | τ = 3, α = 0,4: var(i)/var(e) **2,552 (k = 1) / 3,415 (k = 2) / 4,403 (fest)**, var(u)/var(e) 0,408 / 0,546 / 0,705. τ = 12, α = 0,13: 9,61 / 13,70 / 18,07 / 34,9 / 103,1 (k = 1, 2, 3, 6, fest). **In allen 403 stabilen Zellen** (β = 0, α von 0,01 bis 0,99 α_krit(fest) im Schritt 0,01; τ = 3: k = 1, 2; τ = 4: k = 1..3; τ = 6: k = 1..3; τ = 12: k = 1, 2, 3, 4, 6, jeweils gegen fest; mit jedem k unter τ: 537 von 537, Zählung der unabhängigen Prüfung, nicht im Test) sind var(i) und var(u) der streuenden Lieferzeit kleiner; gilt nur für das anteilige Modell. Gegenproben: Summe der Quadrate der Impulsantwort gleich der Lyapunov-Lösung (relativ 10⁻⁶, 6 Fälle), Simulation innerhalb 5 % (3 Fälle) | `test_noise_variances_tau_3_alpha_04_beta0`, `test_noise_variances_tau_12_alpha_013_beta0`, `test_smaller_bestand_variance_in_every_stable_cell_of_the_whole_alpha_raster`; Oracle: `test_lyapunov_equals_sum_of_squared_impulse_response`, `test_lyapunov_equals_simulation_within_a_band` |
| Und mit voller Leitungsrechnung (β = 1)? | Grenze α = 2 für jede Verteilung (Gegenprobe an 60 zufälligen Verteilungen mit Mittel 3 bis 6, je α = 0,3; 1,5; 2,2: Spektralradius abs(1 − α) bis auf 0,03); var(u)/var(e) = α/(2 − α) (α = 0,55: **0,379**), gleich für jede Lieferzeit: in 180 Fällen (dieselben 60 Verteilungen, α = 0,2; 0,9; 1,7) größte Abweichung unter 10⁻⁸. Die Bestandsvarianz hängt an der Verteilung: τ = 3, α = 0,55: **2,409 / 2,912 / 3,254** (k = 1 / 2 / fest); τ = 12, α = 0,3: 8,32 / 9,92 / 10,72 / 11,90 / 12,96 (k = 1, 2, 3, 6, fest); in allen 1 194 Zellen (α von 0,01 bis 1,99) kleiner als fest. Bestandsposition: IP′ = (1 − α)·IP − e in Brüchen exakt (4 Ketten × 10 Schritte) | `test_noise_variances_beta1_order_variance_is_distribution_free`, `test_beta_one_bestand_variance_smaller_and_order_variance_equal_in_every_cell`; Oracle: `test_beta_one_order_variance_for_arbitrary_distributions`, `test_beta_one_limit_is_two_for_arbitrary_distributions`; Hand: `test_hand_inventory_position_identity_in_fractions` |
| Einzelbestellung: Streuung bei konstantem Bedarf | σ = 0, D = 10, τ = 3, β = 0 (exaktes zweites Moment): var(i)/D² bei α = 0,2: **1,288 (k = 1), 0,773 (k = 2), 0 (fest)**; bei α = 0,4: 1,5625 und 1,1997; hängt nicht an D (D = 1, 10, 100 gleich bis 10⁻⁹). Simulation (800 Läufe) innerhalb 4 % des exakten Werts | `test_single_order_variance_at_constant_demand_sigma_zero`; Oracle: `test_exact_second_moment_equals_simulation_within_a_band` |
| Einzelbestellung: Grenze des zweiten Moments | Niedriger als die des Mittelwerts: τ = 3: k = 1: **0,872** (Mittelwert 10), k = 2: 0,678 (0,889), fest 0,618; τ = 12: k = 2: **0,238** (0,306), k = 3: 0,188 (0,206), k = 6: 0,151 (0,154), k = 1: 0,416 (46); τ = 24, k = 1: 0,291. Zwischen den Grenzen ist der Mittelwert stabil und das zweite Moment divergiert (geprüft für τ = 3, k = 2). Bei β = 1 liegt sie in allen 15 geprüften (τ, k) mit k ≤ 8 bei 2,000. Gegen die Raster-Suche über den Spektralradius des Operators: Abweichung höchstens ein Rasterschritt (0,01) | `test_second_moment_limits_are_below_the_mean_limits`, `test_second_moment_limit_with_leitung_is_two_in_all_checked_pairs`; Oracle: `test_second_moment_limit_values_of_the_exact_system`, `test_second_moment_diverges_between_the_limits_and_stays_finite_below` |
| Einzelbestellung mit Bedarfsrauschen | σ = 1, D = 10, β = 0 (exakt): τ = 3, α = 0,4: var(i) **159,4 (k = 1) / 124,0 (k = 2) / 4,40 (fest)** gegen 2,55 / 3,41 / 4,40 anteilig, also das 62fache (k = 1) und das 36fache (k = 2); τ = 12, k = 2, α = 0,1: **652 gegen 13,4**. Deutung (nur an D = 10 und σ = 1 gemessen): Die Lieferzeit-Streuung wirkt mit D², das Rauschen nur mit σ². In allen 161 Zellen (σ = 1, D = 10, α von 0,01 bis 0,99 α_krit(fest), τ = 3 und 12) liegt die Einzelbestellung über der festen Lieferzeit | `test_single_order_with_noise_numbers`, `test_single_order_variance_larger_than_fixed_in_every_cell` |
| Folgt der Mittelwert der Einzelbestellung dem anteiligen Modell? | Stoß e₀ = 1 bei Bedarf 0, 8 000 Läufe, 40 Perioden: in keiner der Perioden, in denen der Standardfehler messbar ist (über 1e-5), mehr als 4 Standardfehler Abstand zur anteiligen Impulsantwort (3 Fälle, β = 0 und 1; der Plan nennt höchstens 3,24 Standardfehler bei 20 000 Läufen). Mit p = 1 sind beide Modelle Pfad für Pfad gleich (1e-9) | Oracle: `test_mean_of_the_individual_order_follows_the_fluid_impulse_response`, `test_individual_order_with_p_one_is_exactly_the_fluid_model` |
| Stabil heißt nicht schnell | τ = 12, α = 0,2, β = 0: Abklingzeit auf 1 % bei k = 1: **106**, k = 2: **300**, k = 3: **3 193 Perioden** (ρ = 0,9574 / 0,9854 / 0,9986); k = 6 und fest instabil (ρ = 1,0147 / 1,0241). τ = 3, α = 0,6: fest ρ = 0,9914, 542 Perioden; k = 2: 0,9176, 51; k = 1: 0,8165, 23. Handbeispiel α = 0,7: fest 1,0369 (instabil), k = 2: 0,9476, k = 1: 0,8165. Die Schätzung ln 0,01/ln ρ trifft die gemessene Zeit in 5 Fällen auf höchstens 6 Perioden oder 20 % | `test_decay_time_tau_12_alpha_02_beta0`, `test_decay_time_tau_3_alpha_06_beta0`, `test_hand_example_alpha_07`; Oracle: `test_decay_time_against_log_estimate_within_a_band` |
| Und der beste Spektralradius über α? | Bei fester Lieferzeit der **kleinste**: τ = 3: fest **0,6667** (= (L − 1)/L, Doppelwurzel bei α = (1/L)((L − 1)/L)^(L−1)), k = 2: 0,7287, k = 1: 0,8165 (= √q); τ = 12: fest 0,9167, k = 6: 0,9236, k = 3: 0,9343, k = 2: 0,9422, k = 1: 0,9574. k = 1: ρ = √(1 − 1/τ) auf dem ganzen gerechneten Bereich (τ = 3: α von 0,2 bis 1,9 in 18 Rasterpunkten; τ = 12: α von 0,05 bis 0,3 in 26) | `test_smallest_spectral_radius_over_alpha`, `test_k1_spectral_radius_is_sqrt_q_on_a_wide_alpha_range`; Oracle: `test_best_spectral_radius_of_the_fixed_delay_is_the_double_root` |
| Mit Leitungsrechnung (β = 1): Abklingzeit | τ = 12, α = 0,3: ρ = max(abs(1 − α), q) = 0,917 (k = 1), 0,833, 0,75, 0,70, 0,70 (fest); Abklingzeit **56 / 41 / 34 / 28 / 24 Perioden** | `test_decay_time_with_leitung_tau_12_alpha_03`, `test_beta_one_spectral_radius_is_max_of_one_minus_alpha_and_q` (in `test_eigenschaften_…`) |
| Das Handbeispiel (k = 2, p = 2/3, Mittel 3) | α_krit = p(2 − p) = **8/9**; bei α = 8/9 ist das Polynom (z² − (14/9)z + 1)(z − 1/9), Wurzeln (7 ± 4√2·i)/9 mit Betrag 1 und 1/9, Periode 2π/arccos(7/9) = **9,24**; feste Lieferzeit 3: (√5 − 1)/2 = 0,618, Periode 10; Verhältnis 1,438. k = 1, p = 1/3: α_krit = **10** (Wurzel −1, Periode 2), 16,2mal die feste Grenze. Die Lieferzeit-Verteilung der Kette (4/9, 8/27, 4/27, …) in Brüchen exakt | `tests/test_dynamics.py`: `test_hand_*` |

Die Zahl der Zellen steht in jeder Zählung neben der Aussage; „im Schritt 0,01“ gilt für alle Zählungen über α. Wo „exakt“ steht, ist die Rechnung deterministisch (Lyapunov, Operator, Wurzeln), kein Stichprobenwert.

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen der Vormessung (`rk_v18_streuung.py`, `rk_v18b_einzelbestellung.py`, `rk_v18c_zweites_moment.py`, `rk_v18d_belege.py` der Linienplanung) waren Erwartungen; im Bau mit dem Zufallsgenerator der Demo neu gemessen, mit diesen Abweichungen:

- **Zufallsverteilungen:** Die Vormessung nannte „1 000 von 1 000 über der festen Grenze, Median 1,36, Wiederkehr in 6“ (Dirichlet-Gewichte, numpy-Zufallsgenerator). Im Bau (Exponentialgewichte, SplitMix64): **999 von 1 000 über, 1 innerhalb 10⁻⁶ (fast fest), Median 1,33, Wiederkehr in 4**. In den 1 000 Zufallsverteilungen lag die Grenze in keinem Fall unter der festen (0 von 1 000, dazu 0 von 250 ersten Verteilungen bei den Mitteln 3, 5, 6); ein Beweis für beliebige Verteilungen ist das nicht; „1 000 von 1 000“ ist korrigiert.
- **k = 1:** Die Vormessung schrieb „keine Grenze bis α = 2,6“ (Ende des Rasters); die Wurzelortskurve gibt die Grenze für jedes α: 4τ − 2. „k = 1 nie instabil“ aus der Gliederung gilt diskret nur bis α = 4τ − 2.
- **Kleinster Spektralradius über α:** Die Vormessung (Raster 0,005) nannte 0,669 für fest bei τ = 3 und 0,9215 bei τ = 12; mit lokaler Verfeinerung des Rasters ist es die Doppelwurzel, **0,6667 und 0,9167**; k = 2: 0,7287 statt 0,731; τ = 12: 0,9236 / 0,9343 / 0,9422 statt 0,929 / 0,936 / 0,944. Die Rangfolge (fest am kleinsten) bleibt.
- **β = 1 mit α = 0,55 statt 0,5:** Der Plan rechnete mit α = 0,5 (eine Zweierpotenz). Die Presets und Tests meiden Zweierpotenzen; die Zahlen 2,409 / 2,912 / 3,254 (α = 0,55) ersetzen 2,533 / 3,019 / 3,333 (α = 0,5). Die Bestellvarianz α/(2 − α) und die Grenze 2 sind davon unabhängig. β = 1 selbst ist die Modellwahl „Leitung voll mitrechnen“ (strukturell, nicht eingestellt); p = 1 ist die feste Lieferzeit.
- **Zweierpotenz-Fälle von p:** p = k/τ ist abgeleitet und bei einzelnen Reglerstellungen eine Zweierpotenz (k = 3 und k = 6 bei τ = 12, k = 2 bei τ = 4); die Rechnung ist linear, und `test_dyadic_p_gives_the_same_results_with_rounding_noise` zeigt: Grenze, Spektralradius und Abklingzeit ändern sich bei p·(1 ± 10⁻⁹) nicht. Kein Preset hat eine Zweierpotenz für α oder p.
- **Presets:** „Dritte Ordnung“ nutzt τ = 20 (p = 3/20) statt τ = 12 (p = 1/4); die +51 %- und +66 %-Zahlen stehen im Verdikt der Grenzkarte, unabhängig vom Preset. „Einzelbestellung“ zeigt die Zeile τ = 12, k = 2, α = 0,1 (652 gegen 13,4); die Zeile τ = 3, α = 0,4 (124 gegen 3,41) steht im Verdikt von „Schwankung“.
- **Simulation:** Die Vormessung simulierte 400 000 Perioden in einem Lauf; die App rechnet 200 Läufe zu je 2 000 Perioden, die ersten 400 zählen nicht (anteilig) bzw. bis 600 Läufe zu 1 500 Perioden (Einzelbestellung), die Tests vergleichen mit Bändern von 4 bis 5 %. Die Zahlen der Tabelle sind exakte Rechnungen (Lyapunov, zweites Moment), die Simulationen nur Gegenproben.
- **Gliederung §6a-3 „die Schwankung ist nicht unbedingt kleiner“:** gemessen: im anteiligen Modell ist sie bei gleichem α in allen gezählten Zellen kleiner; „nicht kleiner“ gilt erst für die Einzelbestellung und ist dort eine Schwankung allein aus der Lieferzeit, proportional zu D². Das schließt die Frage der Gliederung mit einem anderen Maß als dem eines Suchtreffers (arXiv 1105.6311, nicht gelesen).
- **Fehler der eigenen Vorrechnung** (aus dem Plan übernommen): Die erste Fassung des Polynoms ließ den Faktor z aus (ergab Lieferzeit k + 1); Zustandsmatrix und Polynom (z − 1)(z − q)^k + α·p^k·z stimmen auf 10⁻¹⁴ (`test_charpoly_equals_matrix_polynomial`, 70 Fälle, größte Abweichung unter 10⁻¹²).

## Wo die Annahmen enden

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Linear, ohne Klemmung** (die Bestellung darf negativ werden) | Mit Bestellung ≥ 0 bleibt die Schwingung hinter der Grenze beschränkt (Grenzzyklus, bei fester Lieferzeit auch Chaos); mit streuender Lieferzeit hier nicht gerechnet | [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) (feste Lieferzeit, β = 0) |
| **Anteilig (Erwartungswert-Modell)** | Kommt jede Bestellung als Ganzes, liegt die Grenze niedriger als im anteiligen Modell (Experiment 4) und das Lager schwankt auch bei konstantem Bedarf | Experiment 4 |
| **Lieferzeit = Summe geometrischer Zeiten** (eigene Wahl) | Andere Formen gehen in die gemessenen Fälle von Experiment 2; ob echte Lieferzeiten so verteilt sind, ist nicht untersucht | – |
| **Kein Meldeverzug** | Meldeverzug zusammen mit Streuung ist nicht gerechnet | [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) |
| **β nur 0 oder 1** | Zwischenwerte nicht gerechnet; bei fester Lieferzeit mit Schwelle bei β = ½ untersucht | [`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo) |
| **Weißes Rauschen, Normalverteilung** | Saison und Resonanz (Frequenzgang) nicht gerechnet | [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo) |
| **Feste Lieferzeit als Grenzfall der Familie** | Die feste Verzögerung in stetiger Zeit (a·τ = π/2) steht in [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo); der stetige Erlang-Kreis hier nur als Vergleichslinie | [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) (feste Lieferzeit) |

## Ehrliche Grenzen

- **Literatur:** Zu Lieferzeit-Verteilungen in Bestandsregeln gibt es Fachliteratur (Suchtreffer); **keine Quelle wurde im Wortlaut gelesen**, auch nicht die Dokumentation der Systemdynamik-Software. Die Formeln sind „eigene Herleitung in diesem Modell“; die Neuheit wird nicht behauptet. Die Verzögerung dritter Ordnung (k = 3) steht für die Standardverzögerung der Software nur nach Suchtreffern.
- **Modellwahl:** Die Summe geometrischer Zeiten ist eine eigene Wahl; ob echte Lieferzeiten so verteilt sind, ist nicht untersucht, und es gibt keinen Vergleich mit Lieferzeitdaten. Alle Zahlen gelten für β = 0 oder β = 1, ohne Klemmung, ohne Meldeverzug.
- **Einzelbestellung:** nur mit der Verteilung „Summe geometrischer Zeiten“ simuliert; Zweipunkt- und Zufallsverteilungen nur im anteiligen Modell. Ein Überholverbot (FIFO) ist nicht gerechnet. Das zweite Moment ist nur für β = 0 und β = 1 untersucht (Zwischenwerte nicht) und in der App nur für k ≤ 8. Die Grenze des zweiten Moments wird auf einem Raster mit 20 % Schritt gesucht.
- **Wiederkehr der Stabilität** (4 von 1 000 Zufallsverteilungen): auf dem α-Raster 0,01 bis 2,6 gefunden, nicht näher untersucht; schmale Fenster zwischen Rasterpunkten sind nicht ausgeschlossen.
- **Zufallsverteilungen:** Die Familie (Exponentialgewichte, auf das Mittel geneigt, 1 bis τ + 4) ist eigene Wahl; „999 von 1 000“ gilt für sie bei Mittel 4, die anderen Mittel sind in den ersten 250 Verteilungen neu gerechnet, der Rest steht nur in der vorgerechneten Reihe.
- **Rauschen:** Das Bedarfsrauschen ist weiß und normalverteilt; die Deutung der 36- bis 62fachen Varianz (D² gegen σ²) ist nur an D = 10 und σ = 1 gemessen.
- **Nicht mit echtem Mausklick getestet:** die Regler und Auswahlen der App sind mit AppTest durchlaufen, nicht mit dem Zeigegerät in jedem Browser.

## Verifikation

- **Kern von Hand:** das Handbeispiel (k = 2, p = 2/3, Mittel 3) in Brüchen: Lieferzeit-Verteilung, Grenze 8/9, Wurzeln, k = 1 mit Grenze 10, Hurwitz-Grenzen 4 und 8/3, Bestandsposition bei β = 1, Bestellvarianz 3/7 (`tests/test_dynamics.py`).
- **Orakel** (`tests/test_oracle_streuende_lieferzeit.py`): feste Lieferzeit mit ausdrücklicher Bestellhistorie gegen die Kette (p = 1); Verteilungsmatrix mit negativer Binomialverteilung gegen die Kette (Zeitreihen); Grenze über Eigenwerte (Raster und Bisektion) gegen die Wurzelortskurve, auch für Zweipunkt-, Gleichverteilung und Zufallsverteilungen; Summe der Quadrate der Impulsantwort und Simulation gegen die Lyapunov-Gleichung; Moment-Operator durch Probevektoren gegen die Kronecker-Fassung; exaktes zweites Moment gegen Simulation; Mittelwert der Einzelbestellung gegen die anteilige Impulsantwort; Abklingzeit gegen ln 0,01/ln ρ; Doppelwurzel der festen Lieferzeit.
- **Eigenschaften** (`tests/test_eigenschaften_streuende_lieferzeit.py`): Mittel k/p; Grenze wächst mit der Streuung in der gemessenen Familie; Grenze der festen Lieferzeit für L = 1 bis 40; Skalierung mit D²; p·(1 ± 10⁻⁹) ändert nichts; β = 1: ρ = max(|1 − α|, q); Annäherung an den stetigen Wert; gleicher Zufalls-Seed gleiche Reihe.
- **Gegenproben für Absolutaussagen:** 1 000 Zufallsverteilungen (Mittel 4) und je 250 bei Mittel 3, 5, 6; 60 zufällige Verteilungen für α/(2 − α); Zählungen über das ganze α-Raster (403, 1 194, 161 Zellen).
- **Claims** (`tests/test_claims.py`): jede Zahl der Befund-Tabelle; Presets: vollständig, Werte innerhalb der Reglergrenzen, keine Zweierpotenz, je Preset die erwartete Verdikt-Zeile (`tests/test_presets.py`); Verdikte an handgesetzten Werten (`tests/test_verdicts.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jede Option der Auswahlen, Eckfälle gleichzeitig, jedes Preset, Permalink), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, Änderung jedes Reglers hebt die Markierung auf), Footer, Laufzeit-Importe.
- **Testzahl:** 223 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `data/slz_frozen.json` | vorgerechnete Reihe der Zufallsverteilungen (Mittel 3, 4, 5, 6) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, pandas, plotly |
| `slz_constants.py` | Regler-Grenzen, feste Messbedingungen, Farben, Presets |
| `slz_dynamics.py` | Kern: Kette, Wurzelortskurve, Lyapunov, Abklingzeit, Simulation, zweites Moment |
| `slz_evaluation.py` | Auswertungen der Experimente, Verdikte (reine Funktionen), Zufallsverteilungen der vorgerechneten Reihe |
| `slz_layout.py` | Fester Kopf (Stylesheet) |
| `slz_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `slz_rng.py` | SplitMix64 (Kopie aus `lgc_rng.py` der lager-chaos-demo), `child_seed`, geometrische Zufallszahlen |
| `slz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/make_frozen.py` | erzeugt `data/slz_frozen.json` |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Claims, Verdikte, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe, Referenzrechnungen (`refimpl.py`) |

## Bewusst nicht umgesetzt

- **Kostenmodell, Sicherheitsbestand, Reglerwahl:** Die Linie erklärt den Kreis, sie entwirft keinen Regler.
- **Meldeverzug und Streuung zusammen**, **Bestellung ≥ 0 (Klemmung, Chaos) mit Streuung**, **Frequenzgang (Resonanz bei streuender Lieferzeit)**: nicht gemessen; die Nachbarn der Linie behandeln Meldeverzug und Frequenzgang einzeln.
- **Zwischenwerte von β** und Fehleinschätzung der Lieferzeit-Verteilung (Drift): nicht gerechnet.
- **Vergleich mit realen Lieferzeitdaten**, **PDF-Export** und **Eigenwerte langer Verteilungsmatrizen**: nicht Teil dieser Demo.
- **Zweites Moment für k > 8** in der App: zu teuer für die Live-Rechnung; die Simulation steht trotzdem da.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).

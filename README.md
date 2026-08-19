# Variance-Gamma model: cena evropske opcije

Implementacija seminarskog rada *„Variance-Gamma model i njegova primena"*
(Matematički fakultet, Univerzitet u Beogradu, 2026): cena evropske kupovne
opcije u VG modelu preko **Esscher transformacije**, sa dve nezavisne rute do
iste cene — **Furijeove inverzije** karakteristične funkcije i **Monte Carlo**
simulacije — ocenjena na realnim tržišnim podacima.

Kod je na engleskom, izveštaji i figure u `outputs/` takođe (radi direktnog
uključivanja u LaTeX); ovaj README je na srpskom.

**Autorstvo.** Rad je formalno koautorski (Vasilije Ivanović, Dejana
Miladinović). Tekst rada, izbor modela i izvođenja su moji; ovaj repozitorijum je
moja samostalna implementacija, napisana od nule i nezavisna od bilo kog koda
priloženog uz rad.

---

## Šta je urađeno

| # | Zadatak | Gde |
|---|---|---|
| 1 | Procena parametara VG procesa i Black-Scholes-a iz istorijskih prinosa (metod momenata i MLE) | [`vg/estimation.py`](src/vg/estimation.py), `scripts/03` |
| 2 | Cena opcije iz parametara preko **inverzne Furijeove transformacije** Esscher-transformisanog procesa: `h*` iz kvadratne jednačine, pa numerička integracija | [`vg/pricing.py`](src/vg/pricing.py), [`vg/esscher.py`](src/vg/esscher.py) |
| 3 | Ista cena preko **Monte Carlo** estimacije — u dve varijante | [`vg/pricing.py`](src/vg/pricing.py) |
| 4 | **Validation set** od realnih evropskih opcija; za svaki input procena parametara, poziv obe funkcije, poređenje sa stvarnom cenom, MSE/RMSE/MAE/MAPE | `scripts/05`, [`outputs/validation_out_of_sample.md`](outputs/validation_out_of_sample.md) |
| 5 | Riziko-neutralna kalibracija na cene opcija, ocenjena i **van uzorka** | `scripts/04`, `scripts/06` |

---

## Brzi start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe scripts\01_fetch_data.py          # skidanje i keširanje podataka
.\.venv\Scripts\python.exe scripts\02_validate_numerics.py   # numerička validacija, bez podataka
.\.venv\Scripts\python.exe scripts\03_physical_estimation.py # fizički parametri iz prinosa
.\.venv\Scripts\python.exe scripts\04_calibrate.py           # kalibracija na cene opcija
.\.venv\Scripts\python.exe scripts\05_validation.py          # predviđanje iz prinosa + metrike
.\.venv\Scripts\python.exe scripts\06_calibration_split.py   # kalibracija ocenjena van uzorka

.\.venv\Scripts\python.exe -m pytest -q                      # 230 testova, ~2 min
```

Rezultati (markdown izveštaji, CSV tabele, PNG + PDF figure) idu u `outputs/`.
Snimak podataka je već u `data/`, pa se skripte 02–06 pokreću i bez koraka 01.

**Vreme izvršavanja.** Skripte 01–03 i 05–06 traju po nekoliko minuta. Skripta 04
traje oko 25 minuta jer fituje i globalno i posebno po svakom od 12 rokova
dospeća, uz multi-start.

**Memorija.** Kvadratura drži privremene nizove oblika `(broj strajkova × broj
čvorova)`; blok je podešen na ~400k elemenata, a `_segment_integral`
prepolovljava blok i pokušava ponovo ako alokacija padne, pa duga kalibracija
preživi i na opterećenoj mašini.

Minimalan primer:

```python
import numpy as np
from vg import VGParams, vg_call_esscher_fourier, vg_call_mc_direct

p = VGParams(theta=-0.14, sigma=0.12, nu=0.20)
K = np.array([90.0, 100.0, 110.0])

vg_call_esscher_fourier(S0=100.0, K=K, r=0.03, T=0.5, p=p)   # Teorema 4.1
vg_call_mc_direct(100.0, K, 0.03, 0.5, p, 10**6, np.random.default_rng(0))
```

---

## Struktura

| Modul | Sadržaj | Deo rada |
|---|---|---|
| [`vg.process`](src/vg/process.py) | VG proces: MGF, karakteristična funkcija, kumulanti, egzaktna simulacija, gustina u zatvorenoj formi (Beselova `K`) | 3.1–3.3, Lema 3.1 |
| [`vg.esscher`](src/vg/esscher.py) | Esscher transformacija, `h*` iz kvadratne jednačine, eksplicitni Q-parametri | 4.2, Lema 4.1, Tvrđenje 4.1 |
| [`vg.pricing`](src/vg/pricing.py) | Gil-Pelaez Furijeova inverzija, Monte Carlo, kvadratura po gustini, Black-Scholes i implicirana volatilnost | 4.3–4.5, jed. (5)–(8) |
| [`vg.estimation`](src/vg/estimation.py) | Fizički parametri iz prinosa: metod momenata i MLE | — |
| [`vg.calibration`](src/vg/calibration.py) | Riziko-neutralna kalibracija na lanac opcija | — |
| [`vg.data`](src/vg/data.py) | Tržišni podaci, čišćenje, forward i diskont iz put-call pariteta | — |
| [`vg.plotting`](src/vg/plotting.py) | Stil figura | — |

**Tri nezavisne rute do iste cene** — Furijeova inverzija, kvadratura po
zatvorenoj formi gustine, i Monte Carlo — dele samo objekat parametara. To je
osnovna provera ispravnosti i sve tri se slažu na `5.4e−07` (nalaz B).

---

## Podaci: `^XSP`, ne `SPY`

Rad vrednuje **evropsku** opciju (Teorema 4.1), a `SPY` opcije su **američke** —
model i podaci bi opisivali različite ugovore i morala bi se argumentima
odbacivati premija za rano izvršenje. Zato je lanac `^XSP` (mini-S&P 500 indeks):
evropske, gotovinski poravnate opcije na indeks.

Snimak `2026-08-18`: **2 987** sirovih ugovora → **617** kupovnih opcija kroz
**12 rokova dospeća** posle filtriranja, **100%** cena su sredine dvostranih
kotacija. Istorija prinosa se uzima sa `^GSPC` (isti proces, mnogo duži uzorak;
dnevni log-prinosi koreliraju `0.99933`, godišnje volatilnosti se slažu na 0.03
procentna poena).

Umesto pretpostavljanja stope i dividendnog prinosa, forward `F` i diskontni
faktor `D` se za svaki rok izvlače iz put-call pariteta, `C(K) − P(K) = D(F − K)`.
Sve formule su generalizovane na trošak držanja `r − q`; za `q = 0` se svode
tačno na formule iz rada.

Izvor je `yfinance` (besplatno, bez API ključa). Svako preuzimanje se upisuje u
`data/` sa vremenskom oznakom, a sve nizvodno čita keširani snimak, pa rezultati
ostaju reproducibilni i kad se tržište pomeri.

---

## Šta je dodato u odnosu na rad

### 1. Esscher-transformisan VG proces je opet VG proces

Iz `D_{h+iu} = D_h − iuν(θ + σ²h) + ½σ²νu²` sledi da pod `Q^h` proces `X` ostaje
VG, sa

```
θ' = (θ + σ²h) / D_h,     σ' = σ / √D_h,     ν' = ν.
```

Ovo apstraktnu promenu mere pretvara u eksplicitnu reparametrizaciju. Korist je
dvostruka: opcija se može simulirati **direktno pod Q** umesto ponderisanja
P-putanja, i dobija se potpuno nezavisna provera Furijeovog koda. Koliko to
vredi — nalaz C.

### 2. Ispravka jednačine (4)

Rad piše `e^{rt} = M(h*+1, t)/M(h*, t)` i zatim `H(h*) = e^{rt}`. Kako je
`M(h,t) = D_h^{−t/ν}`, uslov je zapravo

```
e^{rt} = (D_{h*}/D_{h*+1})^{t/ν}   ⟺   H(h*) = e^{rν}.
```

Oblik `e^{rν}` je tačan i to je bitno: `h*` **ne sme** da zavisi od `t`, jer
jedna mera mora da vrednuje sve rokove dospeća, a samo `e^{rν}` verzija ima to
svojstvo. (Rad sam koristi `e^{rν}` dva pasusa kasnije.) Implementirano je
`H(h*) = e^{rν}`; test to proverava za više `t` sa istim `h*`.

---

## Nalazi

Sve sa referencama na izveštaje u `outputs/`, gde su brojevi i dokazi.

### A. Esscher specifikacija ima dva slobodna parametra, ne tri

Najvažniji nalaz, i direktno se tiče Poglavlja 4 rada.

Pod modelom `S_t = S₀ e^{X_t}` (Sekcija 4.1) cena **nema sopstveni drift**, pa
ceo teret martingalnosti pada na zakon od `X` pod `Q`. Uslov
`E^Q[e^{X_t}] = e^{rt}` daje

```
θ_Q + σ_Q²/2 = (1 − e^{−rν}) / ν.
```

Jedna kombinacija riziko-neutralnih parametara je dakle **fiksirana**, a cene
opcija zavise od fizičkih `(θ, σ, ν)` samo preko `(σ_Q, ν)`. Dve posledice, obe
verifikovane numerički u [`outputs/validation.md`](outputs/validation.md):

1. **Fizičko `θ` se ne može identifikovati iz cena opcija.** Pet fizičkih
   skupova parametara sa `θ` od −0.60 do +0.20, konstruisanih da dele isto
   `(σ_Q, ν)`, daju **identične cene do mašinske preciznosti**. Zato
   `vg.calibration` fiksira `θ = 0` i fituje `(σ, ν)` — time se ne gubi ništa, a
   izbegava se optimizacija duž ravnog pravca.

2. **Dostižni osmesi su skoro simetrični.** Pošto je `θ_Q` prikovano blizu
   `−σ_Q²/2` (tačno to kada je `r = 0`), model proizvodi zakrivljenost ali
   gotovo bez nagiba — ispod petine volatilnostnog poena preko 80%–125%
   moneyness-a, bez obzira na fizičko `θ`.

`fig_04_smile_shapes.png` postavlja obe polovine jednu pored druge: četiri
procesa čije se gustine pod `P` vidno razlikuju, i četiri krive implicirane
volatilnosti koje one proizvode — a koje leže jedna preko druge. Sve što ih
razlikuje pod `P` briše uslov martingalnosti na putu do `Q`.

Ovo je svojstvo *specifikacije*, ne VG procesa i ne numerike.

**Potvrda na tržišnim podacima.** Kalibracija na 617 `^XSP` kupovnih opcija
([`outputs/calibration.md`](outputs/calibration.md)):

| model | slob. par. | σ | ν | RMSE cena | RMSE IV | na granici |
|---|---|---|---|---|---|---|
| Black-Scholes, jedna ravna vol | 1 | 0.13590 | — | $6.003 | 425.6 bp | |
| VG / Esscher | 2 | 0.13594 | 0.00100 | $6.000 | 423.9 bp | `ν` |

Tržišni osmeh na ovom snimku opada za **5.6 volatilnostnih poena** u proseku
(raspon 3.5–8.2) između jednog standardnog pomeraja ispod i iznad forward-a —
čist, realan equity skew.

Esscher fit ne pada samo *blizu* Black-Scholes-a — pada **na** njega:
`σ = 0.13594` naspram ravne BS volatilnosti `0.13590`, uz `ν` na donjoj granici.
Razlog je tačno nalaz iznad: simetričan osmeh ne može da se približi monotono
opadajućem, jer podizanje `ν` diže *oba* krila — pomaže na jednoj strani, odmaže
na drugoj. Najbolji kompromis je da se ne savija uopšte. Drugi parametar kupuje
1.7 baznih poena (425.6 → 423.9). Isto važi i po pojedinačnim rokovima: čak i kad
sme da fituje svaki rok posebno, `ν` završava na donjoj granici na **9 od 12**
rokova, a rang-korelacija `ν` sa `T` je `+0.03` — nema vremenske strukture jer
model odbija da uzme zakrivljenost koju `ν` nudi. Ceo teret nosi `σ`, koje raste
od 0.105 na prednjem kraju do 0.186 na zadnjem (rang-korelacija `+0.96`): tako
model sa ravnom volatilnošću kodira osmeh koji ne ume da predstavi.

`fig_07_residuals.png` pokazuje oblik neuspeha: reziduali nisu šum oko nule nego
sistematski nagib po moneyness-u — nedostajući skew koji se probija.

### B. Furijeova inverzija ne konvergira uniformno — i to prijavljuje

Za `T/ν ≤ 1/2` VG gustina je neograničena u nuli, karakteristična funkcija opada
samo kao `u^{−2T/ν}`, i nijedna Furijeova metoda ne konvergira brzo. To je
svojstvo zakona, ne implementacije.

Zamka je da se prvo fiksira tačka odsecanja pa se onda čvorovi racionišu da bi se
do nje stiglo: nedovoljno razlaganje oscilacije ne gubi cifru, nego vraća broj
bez ijedne tačne cifre. Umesto toga integral se akumulira po dijadskim segmentima
`[0,U], [U,2U], [2U,4U], …`, svaki potpuno razložen, a ostatak se procenjuje na
dva nezavisna načina (geometrijski pad doprinosa i granica `|φ(U)|/β`), od kojih
se prijavljuje **manji**. Rutina vraća `QuadInfo` sa `converged` i
`error_estimate`.

Izmereno ([`outputs/validation.md`](outputs/validation.md)):

| režim | `T/ν` | Furije vs gustina | prijavljena greška |
|---|---|---|---|
| skoro Gausov | 37.5 | `~3e−14` | `1e−13` |
| umereni repovi | 2.5 | `~2e−11` | `3.1e−09` |
| teški repovi | 0.32 | `~3e−07` | `3.6e−05`, `converged=False` |

Najveće neslaganje Furije-vs-gustina preko svih režima: **`5.4e−07`** na aktivi
od $100. Test `test_reported_error_is_an_upper_bound_on_the_real_error` proverava
da je prijavljena greška zaista gornja granica — ceo numerički deo se oslanja na
to.

### C. Jednačina (8) se raspada, a popravka je besplatna

Rad predlaže Monte Carlo preko ponderisanog uzorkovanja (jed. 8): simuliraj pod
`P`, ponderiši sa `exp(h* X_T)/M(h*, T)`. Kako Esscher transformacija slika VG u
VG, alternativa je simulirati direktno pod `Q` sa transformisanim parametrima i
izbaciti težinu. Oba su nepristrasna za istu cenu, ali:

* na sintetičkim testovima ponderisani estimator ima u proseku **5.1×** veću
  standardnu grešku, tj. traži ~**27×** više putanja za istu tačnost
  (`fig_03_mc_convergence.png`);
* na realnim podacima **potpuno otkazuje** kad je `ν` malo. Kratak prozor
  procene daje malo `ν`, malo `ν` gura `h*` daleko (`h* = 0.52` na
  petogodišnjem prozoru, **preko 29** na jednogodišnjem), a težina `exp(h* X_T)`
  postaje lognormalna sa ogromnom varijansom: šačica putanja nosi svu masu.
  RMSE skače na **$21.53** naspram **$6.79** za direktnu simulaciju pod `Q`.

Ista mera, ista cena, upotrebljiva varijansa.

### D. Predviđanje iz prinosa: premija za rizik varijanse

Ovo je test koji konstrukcija rada zapravo implicira — proceni pod `P`, pređi na
`Q` Esscher transformacijom, vrednuj. **Nijedna cena opcije se ne fituje**, pa je
svaki broj van uzorka. Validation set: 617 evropskih opcija, 12 rokova
([`outputs/validation_out_of_sample.md`](outputs/validation_out_of_sample.md)).

Najbolji prozor procene — 5 godina (1 260 prinosa):

| metod | MSE | RMSE ($) | MAE ($) | RMSE (vol. poeni) |
|---|---|---|---|---|
| Black-Scholes (MLE iz prinosa) | 16.05 | 4.007 | 3.146 | 4.280 |
| VG / Esscher, Furije | 15.75 | 3.969 | 3.082 | 4.148 |
| VG / Esscher, Monte Carlo pod `Q` | **15.04** | **3.878** | **3.036** | **4.119** |
| VG / Esscher, Monte Carlo jed. (8) | 15.79 | 3.974 | 3.082 | 4.178 |

VG pobeđuje Black-Scholes, ali skromno, i oba modela greše u istom smeru iz istog
razloga:

* volatilnost iz realizovanih prinosa: **16.99%**
* prosečna implicirana volatilnost na novcu: **14.18%**

Taj procep je cela priča. Realizovana volatilnost i volatilnost koju tržište
naplaćuje prosto nisu isti broj — to je **premija za rizik varijanse**, i tačno
zbog nje se u praksi kalibriše na cene opcija umesto da se procenjuje iz prinosa.
Nije mana procenitelja; ovde se meri.

### E. Kalibracija ocenjena van uzorka

Kalibrisan model reprodukuje svoje sopstvene kotacije skoro po konstrukciji, pa
greška **u uzorku** meri koliko je model *fleksibilan*, ne koliko je *tačan*.
Zato je svaki model ovde fitovan na jednom delu površine i ocenjen na delu koji
nije video — a Black-Scholes je **fitovan** na cene opcija pod istim uslovima, a
ne procenjen iz istorije, pa se porede modeli, a ne informacioni skupovi
([`outputs/calibration_split.md`](outputs/calibration_split.md)).

RMSE u volatilnostnim poenima, `trening → test`:

| podela | šta testira | Black-Scholes | VG / Esscher |
|---|---|---|---|
| naizmenični strajkovi | ništa (kontrola) | 4.31 → 4.16 | 4.29 → **4.14** |
| ATM → krila | ekstrapolacija *oblika* osmeha | 3.18 → 5.19 | 2.58 → **4.77** |
| kratki → dugi rokovi | ekstrapolacija *vremenske strukture* | 3.79 → 6.366 | 3.75 → 6.369 |

Drugi parametar kupuje nešto samo kad se ekstrapolira oblik osmeha
(4.77 naspram 5.19). Na vremenskoj strukturi ne kupuje ništa — razlika od 0.003
volatilnostna poena je šum, a `ν` opet završava na granici. Kod Lévy procesa
spljoštenost log-prinosa na horizontu `T` iznosi `3ν/T` i mora da opada kao
`1/T`, dok tržišna opada sporije; ali da bi se to uopšte videlo, `ν` mora da bude
slobodno, a ovde nije.

Ovo je i metodološka poenta. Ako se model fituje minimizacijom kvadratne greške
prema cenama opcija, pa se ista ta greška prema istim tim cenama prijavi kao mera
kvaliteta, i uporedi sa modelom procenjenim iz prinosa — fitovan model pobeđuje
šta god da je. Rezultat je svojstvo postavke, a ne modela. Tri poređenja koja
stvarno nešto znače su razdvojena:

| šta se poredi | na koje pitanje odgovara | gde |
|---|---|---|
| oba procenjena iz prinosa | ko bolje **predviđa**? | `validation_out_of_sample.md` |
| oba kalibrisana, ocenjena u uzorku | ko je **fleksibilniji**? | `calibration.md` |
| oba kalibrisana, ocenjena van uzorka | ko **generalizuje**? | `calibration_split.md` |

Samo prvo i treće su testovi modela.

### F. Metod momenata i MLE se ne slažu na dnevnim podacima

Na 2 510 dnevnih `^GSPC` prinosa: momenti daju `ν = 0.0217`, MLE daje
`ν = 0.0048` — faktor 4.6
([`outputs/physical_estimation_daily.md`](outputs/physical_estimation_daily.md)).

Uzrok je strukturni. Oblik Gama časovnika po koraku je `dt/ν`, ovde 0.18; ispod
`1/2` gustina ima stepeni singularitet u nuli, pa verodostojnost meri koliko
oštro parametri šiljaste u modu, a ne koliko dobro opisuju rasipanje podataka.

Tvrdnja modela o šiljku je proverljiva i **pada**: po momentnom fitu 19.5% dnevnih
prinosa treba da padne unutar `1e−4` od drifta („danas se ništa nije desilo");
u stvarnom `^GSPC` nizu to čini **1.3%**. Tržišta trguju kontinuirano i ne
gomilaju prinose u nuli onako kako Gama časovnik ovog oblika tvrdi. To je realno
ograničenje VG-a kao opisa fizičke mere, odvojeno od toga kako vrednuje opcije
pod `Q`.

Momentne procene se nose dalje, jer cene opcija zavise od varijanse, asimetrije i
spljoštenosti raspodele prinosa, a ne od visine moda. Neslaganje se ne gubi
prosto ređim uzorkovanjem: odnos dve procene `ν` je 4.6× dnevno, 1.6× nedeljno,
3.0× mesečno — nedeljni slučaj se popravlja kako mehanizam predviđa, ali mesečni
niz ima samo 119 opservacija sa uzoračkom spljoštenošću koju određuje šačica
kriznih meseci, pa je prekratak da bi bilo šta rešio.

### G. Put-call paritet na ustajalim kotacijama

Van američkog radnog vremena Yahoo vraća `bid = ask = 0`, pa su jedine dostupne
cene poslednje trgovine — a one su za kolove i putove nastale u različitim
trenucima. Dvoparametarska regresija `C − P = D(F − K)` tu čita šum vremena kao
zakrivljenost: na ranijem snimku davala je diskontne faktore **iznad 1**
(negativne kamate) i dividendne prinose od **−4%** na polovini rokova. Robusna
varijanta (fiksiran `D` iz `^IRX`, `F` kao medijana `K + (C−P)/D` blizu novca) to
uklanja. Obe su implementirane; izbor je automatski prema udelu dvostranih
kotacija, a metod se beleži u koloni `fwd_method` tako da ništa ne ulazi u
kalibraciju na tihoj pretpostavci.

Uzgredna posledica: fitovanje `log F(T) = log S_eff + carry·T` kroz sve rokove
daje efektivni spot **770.33** naspram kotirane poslednje cene **778.58** —
**1.06%** razlike. To nije modelska odluka nego vremenska oznaka: poslednji bar
istorije indeksa i kotacije opcija dolaze iz različitih sesija. Ispočetka je taj
procep obarao *svaki* kratki rok na proveri forward-a, iako su im forwardi bili
odlični (dva nezavisna procenitelja slagala su se na dve hiljadite dolara).
Provera krive prema samoj sebi umesto prema ustajalom spotu ih zadržava.

---

## Numerička validacija

`scripts/02_validate_numerics.py` proverava, **bez ikakvih tržišnih podataka**,
da je implementacija ispravna pre nego što se uperi u tržište:

- **tri nezavisne rute do cene** — najveće neslaganje `5.4e−07` na aktivi od $100;
- **Monte Carlo** — najveći z-skor prema Furijeovoj ceni `1.98` standardnih
  grešaka preko 15 kombinacija strajk/režim;
- **martingalnost** — `E^Q[e^{X_T}] = e^{rT}` do `1e−11` za svako `t`;
- **granicu `ν → 0`** — monotona konvergencija VG → Black-Scholes;
- **identifikabilnost** — tabela iz nalaza A.

Test suite (230 testova) dodatno proverava momente prema Monte Carlu, gustinu
prema kvadraturi (integral `= 1.0000000000` u sva četiri režima, uključujući
singularne), kumulante prema Čebiševljevom razvoju `log M`, put-call paritet sa
nezavisno računatim putom, monotonost i konveksnost po strajku, i granice
odsustva arbitraže.

---

## Literatura

- Madan, Carr, Chang, *The Variance Gamma Process and Option Pricing*, European Finance Review 2(1), 1998.
- Cont, Tankov, *Financial Modelling with Jump Processes*, Chapman & Hall/CRC, 2004.
- Gil-Pelaez, *Note on the inversion theorem*, Biometrika 38(3–4), 1951.
- Shenoy, Kempthorne, *The Variance Gamma Process for Option Pricing*, arXiv:2510.14093, 2024.

---

## Licenca

MIT — vidi [LICENSE](LICENSE).

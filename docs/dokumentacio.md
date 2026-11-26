# Okostelefonos emberi aktivitásfelismerés (HAR) alany-diszjunkt kiértékeléssel és kompakt 1D-CNN architektúrával

**Tantárgy:** Mély gépi tanulás (`NSXMG1HMLF`)  
**Intézmény:** Óbudai Egyetem — Kiberfizikai Rendszerek Intézet  
**Tárgyfelelős / Oktató:** Dr. habil. Dineva Adrienn  
**Szerző:** Tanácsi Roland (Neptun: `HCY41E`) — egyéni projektmunka  
**Félév:** 2026/27/1 (levelező tagozat)  
**Dátum:** 2026. november 25.  

---

## Vezetői összefoglaló

A projekt célja egy **nulláról (from scratch) tanított, kompakt PyTorch 1D konvolúciós neurális hálózat (1D-CNN)** és klasszikus gépi tanulási baseline modellek (**Logisztikus Regresszió**, **Random Forest**) szigorúan tudományos, adat-szivárgásmentes összehasonlítása okostelefonos inerciális szenzorjeleken (UCI HAR adatkészlet, `10 299` ablak, `9` csatorna × `128` időlépés `50 Hz`-en).

Minden kísérlet **szigorúan alany-diszjunkt (`subject-disjoint`)** felosztás mellett zajlott (`16` tanuló-, `5` validációs és `9` lezárt tesztalany, plusz `5`-szörös `StratifiedGroupKFold` keresztvalidáció a `21` tanulóalanyon). A skálázók és statisztikák kizárólag a tanulóhalmazon kerültek illesztésre, és a `9` tesztalanyból álló `2 947` ablakos hivatalos teszthalmaz kizárólag **egyetlen alkalommal**, a végső modellek lezárása után került kiértékelésre.

A végleges teszteredmények igazolják, hogy a hangolt **`CompactCNN1D (48-96-160, k=9-5-3)`** (`74 582` tanítható paraméter, `0,30 ms` GPU / `0,39 ms` CPU ablakonkénti késleltetés) **`0,9283` Macro-F1** (`92,70%` pontosság) értéket ér el a látatlan tesztalanyokon — szignifikánsan felülmúlva mind a Logisztikus Regressziót (`0,9024`, párosított McNemar $p = 4{,}31 \cdot 10^{-5}$), mind a látatlan alanyokon túltanuló Random Forest modellt (`0,8428`, $p = 1{,}61 \cdot 10^{-34}$). A kontrollált csatorna-abláció egyúttal bizonyította, hogy a gravitációs gyorsulási komponens (`total_acc`) elhagyása drasztikusan rontja a statikus testhelyzetek (`LAYING` $1{,}000 \rightarrow 0{,}866$) szeparációját.

---

## 1. Szakirodalmi áttekintés

A viselhető szenzorokon és okostelefonokon alapuló emberi aktivitásfelismerés (*Human Activity Recognition*, HAR) kulcsfontosságú részterülete a pervasív számítástechnikának, az egészségmonitorozásnak és a kiberfizikai rendszereknek. A korai okostelefonos megközelítések — köztük a jelen munka alapjául szolgáló adatkészletet bemutató **Anguita et al. (2012, 2013)** publikációk — a gyorsulásmérő és a giroszkóp háromtengelyű idősoraiból csúszóablakos szegmentálás után több száz kézzel tervezett idő- és frekvenciatartományú statisztikát (átlag, szórás, entrópia, energia, FFT-sávok) számítottak ki, majd azokra klasszikus osztályozókat (például többosztályos SVM-et vagy döntési fákat) illesztettek. Bár ezek a módszerek alacsony számítási igényűek és jól értelmezhetők, teljesítményük erősen függ a szakértői jellemzőtervezéstől, és globális ablakstatisztikákba sűrítve elveszítik a mozgásciklusok finom időbeli fázisszerkezetét.

A mélytanulás elterjedésével (**Bishop, 2023; Prince, 2024**) a kétdimenziós képfeldolgozásból ismert konvolúciós elv az egyváltozós és többcsatornás idősorokra is sikeresen átültetésre került az idődimenzió mentén eltolásinvariáns lokális szűrők segítségével. **Ronao és Cho (2016)** az elsők között igazolták az okostelefonos HAR területén, hogy a többcsatornás idősorokra alkalmazott 1D konvolúciós neurális hálózatok (*1D-CNN*) képesek közvetlenül a nyers gyorsulás- és szögsebesség-jelekből hierarchikus, időbeli és szenzorok közötti reprezentációkat tanulni, felülmúlva a klasszikus kézi jellemzőkészletekre épülő modelleket. **Ignatov (2018)** tovább finomította ezt az irányt: megmutatta, hogy már egy sekély, valós idejű mobil inferenciára méretezett 1D-CNN architektúra is képes hatékonyan kinyerni a lokális mozgásmotívumokat, különösen ha a lokális konvolúciós mintázatokat globális idősor-eloszlási információkkal kombináljuk.

A szakirodalom ugyanakkor rávilágít két gyakori módszertani buktatóra is. Egyrészt az átfedő csúszóablakok (például 50%-os átfedés) véletlenszerű ablak-szintű szétválasztása (*random window split*) ugyanazon alany szomszédos ablakait juttatja a tanító- és teszthalmazba, ami irreálisan felfújt pontosságot eredményez. Ezért a megbízható mobil HAR-értékelés alapfeltétele a szigorú **alany-diszjunkt (*subject-independent*)** kiértékelés. Másrészt a nyers gyorsulásjelekben lévő alacsony frekvenciás gravitációs összetevő eltávolítása ugyan segíti a dinamikus járásmintázatok elemzését, de elrejti a készülék statikus térbeli orientációját, amely a nyugalmi testhelyzetek (`SITTING`, `STANDING`, `LAYING`) megkülönböztetéséhez elengedhetetlen. Jelen projekt kifejezetten e két aspektust vizsgálja reprodukálható, lezárt teszthalmazos protokoll mellett.

---

## 2. Problémafelvetés és kutatási hipotézisek

### 2.1. A megoldandó feladat és kontextus

Adott egy derékra rögzített okostelefon 3 tengelyű gyorsulásmérőjéből és 3 tengelyű giroszkópjából `50 Hz` mintavételezéssel nyert, `2,56` másodperces (`T = 128` minta), `9` csatornás időablak $\mathbf{X}_i \in \mathbb{R}^{9 \times 128}$. A feladat az ablakhoz tartozó $y_i \in \{1,\dots,6\}$ emberi aktivitási osztály (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) automatikus felismerése olyan korábban **nem látott személyeken**, akiknek egyetlen mintája sem szerepelt a tanításban vagy a modellválasztásban. A gyakorlati cél egy olyan kompakt neurális architektúra megalkotása, amely beágyazott/mobil környezetben is szub-milliszekundumos késleltetéssel futtatható, nem támaszkodik külső előtanított alapmodellekre, és pontosan összevethető a klasszikus összegző statisztikás modellekkel.

### 2.2. Kutatási kérdés (RQ) és hipotézisek

- **Kutatási kérdés (RQ):** Szigorúan alany-diszjunkt kiértékelés mellett képes-e egy nulláról tanított, kompakt PyTorch 1D-CNN nyers `(9, 128)` szenzorablakokon magasabb Macro-F1 értéket elérni látatlan alanyokon, mint az ugyanazon `9` csatornából kiszámított ablakstatisztikákra illesztett klasszikus gépi tanulási alapmodellek (`Logistic Regression`, `Random Forest`) — és mely tevékenységeknél vagy alanyoknál jelentkeznek a fennmaradó hibák?
- **Elsődleges hipotézis ($H_1$):** A lokális temporális konvolúciós szűrők megőrzik a lépésciklusok hullámformáját és a csatornák közötti fázisviszonyokat a `2,56 s`-os ablakban, ezért a 1D-CNN szignifikánsan jobb látatlan-alany Macro-F1 eredményt ér el a klasszikus statisztikai modelleknél (különösen a dinamikus lokomóciós osztályokban), miközben kompakt (`< 100k` paraméter) marad.
- **Nullhipotézis ($H_0$):** A globális csatorna-statisztikák tartalmazzák a szeparációhoz szükséges összes információt, így a 1D-CNN nem nyújt érdemi Macro-F1 előnyt a látatlan tesztalanyokon, vagy túltanulja a tanulóalanyok egyedi járásmintázatait.
- **Csatorna-ablációs hipotézis ($H_2$):** A teljes gyorsulási csatorna (`total_acc_xyz`, amely tartalmazza a gravitációs komponenst) elhagyása és a pusztán gravitáció-mentesített jelek (`body_acc_xyz + body_gyro_xyz`) használata drasztikusan rontja a statikus testhelyzetek (`SITTING`, `STANDING`, `LAYING`) felismerését.

### 2.3. Mérési elvek

Mivel a hat osztály enyhén kiegyensúlyozatlan (a legkisebb és legnagyobb osztály aránya `986 / 1407 = 0,701` a tanulóhalmazon), az elsődleges döntési és modellválasztási metrika a **Macro-F1**, kiegészítve az összpontossággal (*Accuracy*), az osztályonkénti F1-értékekkel, a `6×6`-os konfúziós mátrixokkal és az alanyonkénti Macro-F1 eloszlással.

---

## 3. Adatkészlet eredete, struktúrája és auditja

### 3.1. Adatforrás és előfeldolgozási háttér

Az **UCI HAR Using Smartphones** adatkészlet (Anguita et al., 2013) `30` önkéntes (`19–48` év) méréseit tartalmazza, akik derékra erősített Samsung Galaxy S II készülékkel végeztek hat mindennapi tevékenységet. A beépített gyorsulásmérő és giroszkóp `50 Hz`-en rögzítette a 3 tengelyű lineáris gyorsulást (`g` egységben) és a 3 tengelyű szögsebességet (`rad/s`). A szerzők zajszűrés (mediánszűrő + 3. rendű, `20 Hz`-es aluláteresztő Butterworth szűrő) után egy `0,3 Hz`-es Butterworth aluláteresztő szűrővel szétválasztották a teljes gyorsulást (`total_acc`) gravitációs és testmozgási (`body_acc`) komponensre, majd az idősorokat `2,56 s` széles, `50%`-os átfedésű ablakokra (`128` minta/ablak) szegmentálták. A nyers idősorok mellett az eredeti adatkészlet egy `561` elemű előre számított jellemzővektort (`X_train.txt`, `X_test.txt`) is tartalmaz — jelen projektben azonban **nem** ezekre az előre kiszámított származtatott jellemzőkre támaszkodunk, hanem közvetlenül az `Inertial Signals/` könyvtár 9 nyers csatornájával dolgozunk (`body_acc_{x,y,z}`, `body_gyro_{x,y,z}`, `total_acc_{x,y,z}`).

### 3.2. Reprodukálható fájlszintű és tenzor-audit (`src/inspect_data.py`)

Az 1. fázisban elkészített `src/inspect_data.py` szkript lefutása (`outputs/dataset_summary.json`) a teljes adatkészletet közvetlenül lemezről ellenőrizte:

| Tulajdonság | Hivatalos `train` | Hivatalos `test` (lezárva az 5. fázisig) | Összesen / Megjegyzés |
|---|---:|---:|---|
| **Ablakok száma (`N`)** | `7 352` | `2 947` | `10 299` ablak |
| **Nyers tenzor alak és típus** | `(7352, 9, 128)`, `float32` | `(2947, 9, 128)`, `float32` | Mind a `9` csatornafájl hiánytalan |
| **Alanyok száma és ID-k** | `21` fő (`1,3,5,6,7,8,11,14..17,19,21..23,25..30`) | `9` fő (`2,4,9,10,12,13,18,20,24`) | `30` egyedi alany, **átfedés = 0** |
| **Ablak / alany tartomány** | `281 – 409` ablak | `288 – 381` ablak | Kiegyensúlyozott alanyterhelés |
| **Hiányzó / végtelen / duplikált érték** | `0` NaN, `0` Inf, `0` duplikátum | `0` NaN, `0` Inf, `0` duplikátum | Halmazok közötti duplikátum: `0` |

Az egyes tevékenységi osztályok eloszlása az alábbi:

| Osztály ID és név | Hivatalos `train` db (%) | Hivatalos `test` db (%) |
|---|---:|---:|
| `1 — WALKING` | `1 226` (`16,68%`) | `496` (`16,83%`) |
| `2 — WALKING_UPSTAIRS` | `1 073` (`14,59%`) | `471` (`15,98%`) |
| `3 — WALKING_DOWNSTAIRS` | `986` (`13,41%`) | `420` (`14,25%`) |
| `4 — SITTING` | `1 286` (`17,49%`) | `491` (`16,66%`) |
| `5 — STANDING` | `1 374` (`18,69%`) | `532` (`18,05%`) |
| `6 — LAYING` | `1 407` (`19,14%`) | `537` (`18,22%`) |

A csatornánkénti statisztikai audit fontos fizikai különbséget tárt fel: míg a `body_acc_*` csatornák közel zéró várható értékűek (szórás `0,107 – 0,195 g`), a `body_gyro_*` szögsebesség-jelek akár `[-5,97; +5,75] rad/s` tartományban mozognak (szórás `0,256 – 0,407 rad/s`), a `total_acc_x` átlaga pedig `0,805 g` a domináns függőleges gravitációs tengely miatt. Ez közvetlenül indokolja a csatornánkénti standardizálást — szigorúan csak a tanulóhalmazon illesztve.

---

## 4. Szivárgásmentes adatfolyam és modellarchitektúrák

### 4.1. Szubjektum-diszjunkt adatfelosztás és zéró szivárgás

Mivel minden egymást követő időablak 50%-ban átfedi az előzőt, az azonos alanyhoz tartozó ablakok szétválasztása azonnali adatszivárgáshoz vezetne. Ezért a `src/data.py` modul minden felosztási ponton kódszintű `assert_subject_disjointness(...)` ellenőrzést kényszerít ki:

1. **Elsődleges fejlesztői felosztás:** A `21` hivatalos tanulóalany `StratifiedGroupKFold(n_splits=4, val_fold=0, seed=42)` eljárással került kettéosztásra:
   - **`train`:** `16` alany (`[1, 3, 5, 7, 8, 14, 15, 16, 19, 22, 25, 26, 27, 28, 29, 30]`), **`5 563` ablak** (`75,67%`).
   - **`val`:** `5` alany (`[6, 11, 17, 21, 23]`), **`1 789` ablak** (`24,33%`).
2. **Összesített tanulóköri keresztvalidáció:** A `5`-szörös `StratifiedGroupKFold(n_splits=5, seed=42)` mind a `21` tanulóalany (`7 352` ablak) felett ellenőrzi, hogy egy hiperparaméter-választás ne kizárólag a `val_fold=0` öt alanyára hangolódjon rá.
3. **Lezárt teszthalmaz (`test`):** A `9` tesztalany (`[2, 4, 9, 10, 12, 13, 18, 20, 24]`, `2 947` ablak) az 1–4. fázis alatt fizikailag sem került betöltésre (kizárólag a `subject_test.txt` azonosítói kerültek beolvasásra a diszjunkciós állításokhoz).

### 4.2. Klasszikus gépi tanulási baseline modellek (`src/features.py`, `src/train_baselines.py`)

A klasszikus modellek összehasonlításához minden `(9, 128)` ablakból **ablakonként függetlenül** (`axis=-1`) `8` jól értelmezhető időtartományú statisztikát számítottunk ki mind a `9` csatornára (`9 × 8 = 72` jellemző): `mean`, `std`, `min`, `max`, `median`, `iqr`, `rms`, valamint `mean_abs_diff` ($\frac{1}{T-1}\sum_t |x_{t+1}-x_t|$, a jel érdességének / frekvenciatartalmának közelítésére). A két modell:

- **Logisztikus Regresszió (`LogisticRegression`):** `sklearn.pipeline.Pipeline` belsejében alkalmazott `StandardScaler`-rel (szigorúan csak az adott tanuló-foldon illesztve), a validációs Macro-F1 alapján hangolt `C ∈ {0,05; 0,1; 1,0; 10,0}` regularizációval (`max_iter=2000`).
- **Random Forest (`RandomForestClassifier`):** `n_estimators=300` fával, `max_depth ∈ {None, 16, 12}` és `min_samples_leaf ∈ {1, 2}` jelöltek közül a validációs Macro-F1 szerint kiválasztva.

### 4.3. Nulláról tanított PyTorch 1D-CNN architektúrák (`src/models/cnn1d.py`)

A neurális modellek bemenete a szigorúan a tanulóhalmaz `16` alanyán illesztett `(1, 9, 1)` átlaggal és szórással standardizált $\tilde{\mathbf{X}} \in \mathbb{R}^{B \times 9 \times 128}$ nyers tenzor. Két kompakt, jól magyarázható architektúra került implementálásra nulláról:

1. **`CompactCNN1D` (szekvenciális 1D-CNN):** Három temporális konvolúciós blokkból áll (`Conv1d → BatchNorm1d → ReLU → MaxPool1d(2)` az első két blokkban, majd a harmadik `Conv1d → BatchNorm1d → ReLU` blokk után globális átlagoló `AdaptiveAvgPool1d(1)`). A globális időbeli átlagolás megőrzi a gravitációs egyenáramú orientációt (kritikus a `SITTING/STANDING/LAYING` elkülönítéséhez), miközben eltolásinvariáns és drasztikusan csökkenti a paraméterszámot (`Dropout(p)` + lineáris kimeneti fej). A Fázis 3 alapmodell `(32, 64, 128)` csatornaszámmal és `(7, 5, 3)` kernelekkel **`38 054`**, míg a Fázis 4 hangolt változat `(48, 96, 160)` csatornaszámmal és `(9, 5, 3)` kernelekkel **`74 582`** tanítható paraméterrel rendelkezik.
2. **`CompactResCNN1D` (reziduális 1D-CNN):** Három kétkonvolúciós reziduális blokk (`Conv1d → BN → ReLU → Dropout(0,1) → Conv1d → BN` + `1×1` projekciós átkötés) `(32, 64, 128)` szélességgel és `(7, 5, 3)` kernelekkel (**`126 278` paraméter**).

### 4.4. Hardver- és szoftverkörnyezet

Minden futtatás a projekt `.venv` környezetében (`Python 3.11.9`, `PyTorch 2.11.0+cu128`, `CUDA 12.8`, `cuDNN 91900`, `scikit-learn 1.9.1`, `NumPy 2.4.6`) zajlott Windows 11 Pro operációs rendszeren, `Intel Core Ultra 9 285K` (24 szál) CPU, `~63,27 GiB` RAM és `NVIDIA GeForce RTX 5090` (`31,82 GiB` VRAM, compute capability `12.0`) GPU használatával. A véletlenszám-generátorok seedelése és a `cudnn.deterministic=True` beállítás rögzítve lett, ugyanakkor a GPU/CUDA-kernelek közötti bit-szintű determinizmus nem feltételezett.

---

## 5. Kísérletek: Baseline eredmények, hiperparaméter-optimalizálás és ablációk

### 5.1. Fázis 2 klasszikus baseline eredmények (csak `val` és 5-fold GroupCV)

A `src/train_baselines.py` futtatása (`configs/baselines.yaml`, `7,90 s`) az alábbi eredményeket adta a `72` összegző statisztikán:

| Modell | Kiválasztott paraméter (`val`) | Train Macro-F1 | Val Macro-F1 (`5` alany) | Val Acc | 5-Fold GroupCV Macro-F1 (`21` alany) |
|---|---|---:|---:|---:|---:|
| **Logistic Regression** | `C=10.0`, `max_iter=2000` | `0,9725` | **`0,8969`** | **`0,8994`** | **`0,8939 ± 0,0291`** |
| **Random Forest** | `n=300`, `max_depth=None` | `1,0000` | `0,8908` | `0,8983` | `0,8774 ± 0,0410` |

![Klasszikus baseline modellek validációs konfúziós mátrixai](outputs/figures/phase2_baselines_val_cm.png)

A validációs hibaanalízis már ebben a fázisban világos strukturális különbséget mutatott: a **Logisztikus Regresszió** jól szeparálta a dinamikus mozgásokat (`WALKING` `0,9426`, `WALKING_DOWNSTAIRS` `0,9066`), de `104 / 325` `SITTING` ablakot `STANDING`-nek tévesztett (`SITTING` F1 `0,8022`, `6.` alany `0,7757`). Ezzel szemben a **Random Forest** tökéletesen memorizálta a tanulóhalmazt (`1,0000`), és jobban kezelte a `SITTING`/`STANDING` párost (`0,9238`/`0,9339`), viszont erősen gyengélkedett a dinamikus járási osztályokban (`WALKING_UPSTAIRS` F1 `0,7774`), a `21.` validációs alanyon `0,6022` Macro-F1-re esve vissza. A két modell legfontosabb jellemzői (`total_acc_x/y__median/mean/rms` és `body_gyro_x__mean_abs_diff`) igazolták, hogy a gravitációs dőlés és a dinamikus szögsebesség-változás a kulcsjelzők.

### 5.2. Fázis 3: A `CompactCNN1D` MVP teljesítménye (`val` split)

A Fázis 3 alapmodell (`configs/cnn_baseline.yaml`, `AdamW lr=1e-3`, `weight_decay=1e-4`, `dropout=0,3`, `38 054` paraméter) **`27,81 s`** alatt futott le, és a `4.` epochban érte el a legjobb validációs teljesítményt (`early_stopping_patience=15`, megállva a `19.` epochnál):

- **Val Macro-F1: `0,9517`** (`Val Accuracy: 0,9475`) — **`+0,0548`** előrelépés a Logisztikus Regresszióhoz és **`+0,0609`** a Random Foresthez képest ugyanazon az `5` látatlan alanyon!
- **Alanyonkénti validációs Macro-F1:** `0,9552 ± 0,0326` (mind az `5` validációs alanyon verte mindkét klasszikus baseline-t: Subj 6 `0,9513`, Subj 11 `1,0000`, Subj 17 `0,9658`, Subj 21 `0,8990`, Subj 23 `0,9597`).
- **Osztályonkénti F1:** A dinamikus osztályok és a `LAYING` közel hibátlanok (`WALKING` `0,9914`, `WALKING_UPSTAIRS` `0,9899`, `WALKING_DOWNSTAIRS` `1,0000`, `LAYING` `1,0000`), a validációs hibák `94,7%`-a (`89 / 94` ablak) a `SITTING` ↔ `STANDING` tengelyre korlátozódott.

![Phase 3 CompactCNN1D tanulási görbék](outputs/figures/phase3_cnn_baseline_curves.png)

### 5.3. Fázis 4: Hiperparaméter-optimalizálás és 5-szörös GroupCV verifikáció

A Fázis 3 tanulási görbéje megmutatta, hogy a hálózat viszonylag gyorsan (`4.` epoch) éri el a `val` Macro-F1 csúcsát, majd a tanítás előrehaladtával enyhe alany-specifikus túltanulás lép fel. A Fázis 4-ben (`configs/cnn_tuning.yaml`, `src/tune_cnn.py`, teljes futási idő `593,63 s`) **8 tudatosan tervezett konfigurációt** vizsgáltunk meg az elsődleges `val` felosztáson, majd a legjobb jelölteket és a reziduális architektúrát **5-szörös `StratifiedGroupKFold` keresztvalidáción** vetettük össze mind a `21` tanulóalanyon:

| Konfiguráció ID | Architektúra és fő változtatás | Paraméter | Best Ep | Val0 Macro-F1 (`5` alany) | 5-Fold GroupCV Macro-F1 (`21` alany) | OOF Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| `cnn_baseline_ref` | `CompactCNN1D (32-64-128, k=7-5-3)` ref | `38 054` | `4` | `0,9523` | `0,9461 ± 0,0356` | `0,9479` |
| `cnn_reg_smooth` | `dropout=0,4`, `wd=1e-3`, `ls=0,05`, `lr=8e-4` | `38 054` | `4` | `0,9491` | — | — |
| `cnn_aug` | `+ train` zaj (`0,03`) és skálázás (`0,08`), `ls=0,05` | `38 054` | `39` | `0,9480` | — | — |
| `cnn_head_mlp` | `+ Linear(128→64)` MLP fej, `ls=0,05` | `45 926` | `4` | `0,9511` | — | — |
| `cnn_lr_low` | `lr=5e-4`, `dropout=0,35`, `ls=0,05` | `38 054` | `4` | `0,9471` | — | — |
| `rescnn_base` | `CompactResCNN1D (32-64-128)`, `ls=0,05` | `126 278` | `15` | `0,9594` | `0,9482 ± 0,0312` | `0,9496` |
| `rescnn_aug` | `CompactResCNN1D + train` augmentáció, `ls=0,05` | `126 278` | `5` | `0,9590` | `0,9493 ± 0,0327` | `0,9511` |
| **`cnn_wider_k9`** *(GYŐZTES)* | **`CompactCNN1D (48-96-160, k=9-5-3)`, `dropout=0,35`, `lr=8e-4`, `wd=5e-4`, `ls=0,05`** | **`74 582`** | **`17`** | **`0,9725`** | **`0,9563 ± 0,0332`** | **`0,9578`** |

![Phase 4 modell-összehasonlítás az elsődleges validáción és az 5-szörös GroupCV-n](outputs/figures/phase4_hpo_comparison.png)

A végső modellválasztás szigorúan az **5-szörös GroupCV átlagos Macro-F1** alapján történt: a **`CompactCNN1D (48-96-160, k=9-5-3)`** érte el a legmagasabb eredményt (`0,9563 ± 0,0332`, OOF `0,9578`), megelőzve mind a Fázis 3 alapmodellt (`0,9461`), mind a közel kétszer akkora `CompactResCNN1D` változatokat (`0,9482` / `0,9493`). A `k=9` kezdő kernel (`0,18 s` időablak) és a mérsékelt kapacitásbővítés (`48→96→160`) jobban kihasználta a `50 Hz`-es mozgási periódust, miközben a paraméterszám továbbra is kompakt maradt (`74 582`). Megjegyzendő ugyanakkor, hogy a 3 különböző seeden (`42, 123, 2025`) mért elsődleges `val` stabilitásvizsgálat **`0,9561 ± 0,0116`** szórást adott (`0,9725`, `0,9480`, `0,9479`), ami őszintén mutatja, hogy az egyes alanyok közötti validációs eredmény érzékeny az inicializációra. *(Megjegyzés: a `label_smoothing=0,05` miatt a keresztentrópia-veszteség alsó korlátja ~`0,28`-ra emelkedik, ami az optimális osztályozási teljesítmény mellett magasabb nyers veszteségértékekben jelenik meg a görbéken.)*

![A lezárt győztes modell tanulási görbéi](outputs/figures/phase4_tuned_cnn_curves.png)

### 5.4. Kontrollált csatorna-abláció ($H_2$ igazolása)

A `CompactCNN1D (48-96-160, k=9-5-3)` architektúrán végzett 5-szörös GroupCV csatorna-abláció eredményei közvetlenül ellenőrzik a $H_2$ hipotézist:

| Csatorna-konfiguráció | Csatornák száma | 5-Fold GroupCV Macro-F1 | OOF `SITTING` F1 | OOF `STANDING` F1 | OOF `LAYING` F1 |
|---|---:|---:|---:|---:|---:|
| **Teljes 9 csatorna (`Full`)** | `9` | **`0,9563 ± 0,0332`** | **`0,8858`** | **`0,8970`** | **`1,0000`** |
| **6 csatorna (`body_acc + body_gyro`, gravitáció nélkül)** | `6` | `0,9253 ± 0,0236` (`-0,0310`) | `0,8446` | `0,8869` | **`0,8664` (`-0,1336`)** |
| **6 csatorna (`total_acc + body_gyro`, nyers hardver IMU)** | `6` | `0,9486 ± 0,0402` (`-0,0077`) | `0,8701` | `0,8902` | `0,9996` |

![Phase 4 csatorna-ablációs eredmények az 5-szörös GroupCV-n](outputs/figures/phase4_channel_ablation.png)

Az eredmény **egyértelműen igazolja $H_2$-t**: ha eltávolítjuk a `total_acc` gravitációs komponenst, a hálózat képtelen megbízhatóan elkülöníteni a vízszintes fekvést (`LAYING` `1,0000 → 0,8664`) a többi nyugalmi állapottól. Érdekes további megállapítás, hogy a pusztán nyers hardveres jelekkel (`total_acc + body_gyro`, a Butterworth-szűrt `body_acc` nélkül) tanított 1D-CNN **`0,9486`** CV Macro-F1-et ér el (`LAYING` `0,9996`), vagyis a konvolúciós szűrők képesek a gyorsulásjel dinamikus komponensének jelentős részét önállóan kinyerni.

---

## 6. Végső tesztkiértékelés és hibaanalízis

### 6.1. Lezárt teszteredmények (`src/eval_test.py` — `2 947` ablak, `9` látatlan alany)

A 4 végleges, előre lezárt modell **egyetlen alkalommal** került lefuttatásra a hivatalos teszthalmazon (`[2, 4, 9, 10, 12, 13, 18, 20, 24]`). A teszthalmaz szubjektum-diszjunkciója és a mentett checkpoint-skálázók `train`-illesztettsége kódban igazolásra került (`runtime = 10,55 s`):

| Modell | Val Macro-F1 | **Teszt Macro-F1** | **Teszt Pontosság** | **Hibák (`/2947`)** | **Teszt alany-átlag ± szórás (`min – max`)** |
|---|---:|---:|---:|---:|---|
| **Logistic Regression** (`C=10.0`) | `0,8969` | `0,9024` | `90,36%` | `284` | `0,8958 ± 0,0673` (`0,7327 – 0,9656`) |
| **Random Forest** (`300` fa) | `0,8908` | `0,8428` | `84,56%` | `455` | `0,8230 ± 0,1059` (`0,5640 – 0,9307`) |
| **CompactCNN1D** (Phase 3 alapmodell) | `0,9517` | `0,9264` | `92,50%` | `221` | `0,9137 ± 0,0899` (`0,7210 – 1,0000`) |
| **CompactCNN1D** (Phase 4 hangolt győztes) | **`0,9725`** | **`0,9283`** | **`92,70%`** | **`215`** | **`0,9158 ± 0,0925` (`0,7491 – 1,0000`)** |

**Párosított McNemar statisztikai próbák (a `2 947` tesztablakon):**
- **Hangolt 1D-CNN vs. Logisztikus Regresszió:** `174` ablaknál csak a CNN helyes, `105`-nél csak a LogReg ($\chi^2 = 16,57$, egzakt kétoldali **$p = 4{,}306 \cdot 10^{-5}$**) — statisztikailag szignifikáns fölény.
- **Hangolt 1D-CNN vs. Random Forest:** `323` vs. `83` diszkordáns ablak (**$p = 1{,}610 \cdot 10^{-34}$**) — a Random Forest súlyosan túltanulta a tanulóalanyokat (`1,0000 → 0,8428`).
- **Hangolt 1D-CNN vs. Phase 3 Alap 1D-CNN:** `29` vs. `23` diszkordáns ablak (**$p = 0,4885$**) — a hangolt modell `6` hibával kevesebbet vét (`215` vs. `221`), de a két 1D-CNN közötti különbség a látatlan tesztalanyokon nem statisztikailag szignifikáns.

### 6.2. Osztályonkénti és alanyonkénti hibaanalízis

![Phase 5 teszt konfúziós mátrixok mind a négy modellen](outputs/figures/phase5_test_confusion_matrices.png)

Az osztályonkénti és alanyonkénti bontás feltárja a teljesítmény valódi természetét:

| Tevékenység | LogReg F1 | Random Forest F1 | 1D-CNN (Alap) F1 | **1D-CNN (Hangolt) F1** |
|---|---:|---:|---:|---:|
| `WALKING` (`496` ablak) | `0,8887` | `0,7608` | `0,9783` | **`0,9970`** (`496/496` recall) |
| `WALKING_UPSTAIRS` (`471` ablak) | `0,8986` | `0,7587` | **`0,9641`** | `0,9596` |
| `WALKING_DOWNSTAIRS` (`420` ablak) | `0,9049` | `0,8349` | **`0,9870`** | `0,9813` (`420/420` recall) |
| `SITTING` (`491` ablak) | **`0,8533`** | `0,8350` | `0,7966` | `0,7945` |
| `STANDING` (`532` ablak) | **`0,8860`** | `0,8673` | `0,8323` | `0,8412` |
| `LAYING` (`537` ablak) | `0,9830` | **`1,0000`** | **`1,0000`** | `0,9963` (`537/537` recall) |

![Phase 5 osztályonkénti F1 összehasonlítás](outputs/figures/phase5_test_per_class_f1.png)

![Phase 5 alanyonkénti Macro-F1 a 9 látatlan tesztalanyon](outputs/figures/phase5_test_per_subject_f1.png)

A részletes hibaelemzés három kulcsfontosságú mintázatot mutat:
1. **Dinamikus mozgások vs. statikus testhelyzetek:** A hangolt 1D-CNN a három járási osztályban (`0,9596 – 0,9970`) és a fekvésben (`0,9963`) gyakorlatilag hibátlan, messze megelőzve a klasszikus modelleket (`+6%` és `+11%` közötti F1-előny a LogReg-hez képest). Ezzel szemben a `215` összes teszthibából **`174` (`80,9%`) kizárólag `SITTING` ↔ `STANDING` tévesztés** (`103` `SITTING → STANDING`, `71` `STANDING → SITTING`).
2. **Alanyonkénti eloszlás és kiugró alanyok:** A `9` tesztalanyból `6`-on (`4, 12, 13, 18, 20, 24`) a hangolt 1D-CNN **`0,9559 – 1,0000`** Macro-F1-et ér el (a `12.` és `24.` alanyon `1,0000`-t!). Ugyanakkor a **`9.` (`0,7620`) és `10.` (`0,7491`) alanynál** mindkét CNN teljesítménye visszaesik — a `9.` alanyon a Logisztikus Regresszió (`0,8647`) még felül is múlja őket a `SITTING`/`STANDING` osztályokban (`0,8533` / `0,8860` vs. `0,7945` / `0,8412`). Ennek fizikai oka, hogy mozdulatlan ülés és állás közben a dinamikus jelek közel zérók, így a különböző testalkatú vagy eltérő derék-dőlésszögű felhasználóknál a kis (`16` alanyos) tanulóhalmazon betanított konvolúciós szűrők érzékenyebbek a statikus dőlésszög-eltolódásra (*domain shift*), mint egy egyszerű lineáris kombináció a globális `total_acc` átlagokon.
3. **Valósidejű mobil inferencia-költség:** A győztes `CompactCNN1D` (`74 582` paraméter, `~292 KB` FP32 súlyméret) egylépéses késleltetése **`0,299 ms` GPU-n** és **`0,387 ms` CPU-n** ablakonként (`0,083 s` a teljes `2 947` ablakos kötegre GPU-n), ami nagyságrendekkel gyorsabb az `1,28 s`-os ablakfrissítési periódusnál.

---

## 7. Tanulságok, korlátok és továbbfejlesztési irányok

### 7.1. Hipotézisek értékelése és tudományos tanulságok

- **$H_1$ (részben / összességében igazolva):** A kompakt 1D-CNN látatlan tesztalanyokon szignifikánsan (`p = 4,31e-5`) felülmúlja a klasszikus alapmodelleket Macro-F1-ben (`0,9283` vs. `0,9024` és `0,8428`), különösen a temporális hullámformát igénylő dinamikus mozgásoknál (`WALKING` `0,9970`, `WALKING_DOWNSTAIRS` `0,9813`). Ugyanakkor a statikus `SITTING`/`STANDING` határon a lineáris `LogReg` meglepően versenyképes maradt (`0,8533`/`0,8860`).
- **$H_2$ (egyértelműen igazolva):** A `total_acc` gravitációs csatorna nélkül (`6 ch`) a `LAYING` F1 `1,0000`-ról `0,8664`-re zuhant (`-0,0310` Macro-F1 a 5-fold GroupCV-n).
- **Módszertani tanulságok:**
  1. A véletlenszerű ablakfelosztás elkerülése és a `StratifiedGroupKFold` használata elengedhetetlen: a `val_fold=0` `0,9725`-ös csúcsa és a `0,9283`-as végső teszteredmény közötti `~4,4` százalékpontos rés pontosan mutatja a `16` tanulóalanyból fakadó alanyközi varianciát, amelyet a `3`-seedes vizsgálat (`0,9561 ± 0,0116`) és az 5-szörös GroupCV (`0,9563 ± 0,0332`) már a teszt megnyitása előtt előre jelzett.
  2. A Random Forest a `72` összegző jellemzőn tökéletesen (`1,0000`) túltanulta a tanulóalanyokat, ami `0,8428`-as tesztvisszaesést eredményezett — a faalapú modellek sérülékenyek az alanyok közötti eloszlás-eltolódással szemben.

### 7.2. Korlátok és jövőbeli fejlesztési irányok

- **Korlátok:** Az UCI HAR adatkészlet mindössze `30` alanyt tartalmaz, fix derék-elhelyezéssel és szűk életkori sávval (`19–48` év), így a modell közvetlenül nem általánosítható zsebben vagy csuklón hordott telefonokra. A `16` alanyos elsődleges tanulóhalmazon lezárt checkpoint helyett (a teszt lezárása után) éles bevetéshez érdemes lenne a végleges hiperparaméterekkel mind a `21` tanulóalanyon átlagolt/ensemble modellt készíteni.
- **Konkrét műszaki továbbfejlesztés:**
  1. **Hibrid architektúra (Ignatov, 2018 nyomán):** A 1D-CNN globális pool rétegét összekapcsolni a `total_acc` gravitációs dőlésszög-statisztikáival, egyesítve a CNN dinamikus pontosságát a lineáris gravitációs stabilitással.
  2. **3D orientációs (`SO(3)`) adatbővítés:** A tanítás során alkalmazott kismértékű véletlen `3D` tengelyforgatással szimulálható az okostelefon egyéni dőlésszög-eltolódása, ami célzottan csökkenthetné a `9.` és `10.` alanyéhoz hasonló `SITTING`↔`STANDING` hibákat.

### 7.3. Reprodukálhatóság és AI-használati átláthatóság

A teljes kísérleti lánc reprodukálható YAML-konfigurációkkal (`configs/*.yaml`) és moduláris Python-parancsokkal (`src/*.py`), minden futtatási metaadattal az `outputs/*.json` fájlokban rögzítve. A dokumentáció és a kódbázis készítése során mesterséges intelligencia alapú fejlesztői asszisztencia (Antigravity coding agent) került igénybevételre a szkriptek strukturálásához, ellenőrzéséhez és a szövegformázáshoz; minden mérőszám, tenzorméret és grafikon közvetlenül a helyi GPU/CPU futtatások ellenőrzött kimeneteiből származik. *(Adminisztratív megjegyzés: a `CONTEXT.md`-ben rögzített 66% vs. 75%-os megajánlott jegy ponthatár-eltérés miatt a projekt a szigorúbb 75%-os követelményszint teljesítésére törekszik.)*

---

## 8. Irodalomjegyzék

1. **Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).** *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* Proceedings of the 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013), Bruges, Belgium, pp. 437–442.
2. **Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2012).** *Human Activity Recognition on Smartphones Using a Multiclass Hardware-Friendly Support Vector Machine.* International Workshop on Ambient Assisted Living (IWAAL 2012), Lecture Notes in Computer Science, pp. 216–223.
3. **Ronao, C. A., & Cho, S.-B. (2016).** *Human activity recognition with smartphone sensors using deep learning neural networks.* *Expert Systems with Applications*, Vol. 59, pp. 235–244. DOI: `10.1016/j.eswa.2016.04.032`.
4. **Ignatov, A. (2018).** *Real-time human activity recognition from accelerometer data using Convolutional Neural Networks.* *Applied Soft Computing*, Vol. 62, pp. 915–922. DOI: `10.1016/j.asoc.2017.09.027`.
5. **Bishop, C. M. (2023).** *Deep Learning: Foundations and Concepts.* Springer. ISBN `978-3-031-45467-7`.
6. **Prince, S. J. D. (2024).** *Understanding Deep Learning.* Cambridge, MA: MIT Press.

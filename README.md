# hand-gesture-recognition-mediapipe-for-game-input

Rozpoznawanie gestów **dwóch dłoni** z kamery (MediaPipe + prosty klasyfikator MLP), które przesyła zdarzenia o wykonywanych gestach do silnika **Unity** przez **TCP**. Dzięki temu gesty z kamery mogą sterować grą jak zwykłe wejście (input).

Projekt jest forkiem [hand-gesture-recognition-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) (Kazuhito Takahashi, tłumaczenie i ulepszenia: [Nikita Kiselov](https://github.com/kinivi)) i został dostosowany do użycia w grach.

## Co zmieniono względem oryginału

- **Rozpoznawanie dwóch rąk** - klasyfikator dostaje cechy obu dłoni naraz (2 × 21 punktów × (x, y) = 84 cechy), a nie pojedynczej dłoni. Dłonie są ustawiane w stałej kolejności (lewa przed prawą), więc wektor cech jest deterministyczny.
- **Własne gesty** - zamiast domyślnych open / close / pointing model rozpoznaje gesty z gry: `Push`, `Heal`, `Choke`, `Lightning` oraz klasę `inactive`.
- **Odsiewanie pustych ruchów** - gest jest uznawany za aktywny dopiero, gdy:
  1. widoczne są obie dłonie,
  2. różnica wysokości między dłońmi osiąga próg `--height_diff_threshold` (domyślnie 30 px, `0` wyłącza próg),
  3. klasyfikator zwróci klasę inną niż `inactive` z pewnością co najmniej `--min_gesture_confidence`.
- **Integracja z Unity przez TCP** (folder `integrations/`) - pipeline zamienia wyniki klasyfikacji na zdarzenia `gesture_started` i `gesture_ended` i wysyła je jako JSON.
- **Uruchamianie z notatnika** (`integrations/run_pipeline.ipynb`) i lokalny serwer testowy (`integrations/test_server.py`).
- **Model dłoni w formacie MediaPipe Tasks** (`hand_landmarker.task`) zamiast starego API `mp.solutions`.

## Wymagania

- Python 3.10+ (projekt rozwijany na Pythonie 3.14)
- Kamera internetowa
- Zależności z requirements.txt

## Instalacja

```bash
git clone <adres-tego-repozytorium>
cd hand-gesture-recognition-mediapipe-for-game-input

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Struktura projektu

```
├── app.py                          # podgląd z kamery, zbieranie danych treningowych
├── integrations/
│   ├── gesture_pipeline.py         # kamera → klasyfikacja → zdarzenia TCP do Unity
│   ├── event_sender.py             # klient TCP (EventSender), JSON linia po linii
│   ├── send_event.py               # wysyłanie pojedynczego zdarzenia z linii poleceń
│   ├── test_server.py              # prosty serwer TCP do testów bez Unity
│   └── run_pipeline.ipynb          # uruchamianie pipeline'u z notatnika
├── model/
│   ├── hand_landmarker.task        # model wykrywania dłoni MediaPipe
│   └── keypoint_classifier/
│       ├── keypoint_classifier.tflite        # klasyfikator gestów (używany domyślnie)
│       ├── keypoint_classifier.keras         # ten sam model w formacie Keras
│       ├── keypoint_classifier_label.csv     # nazwy gestów
│       ├── keypoint.csv                      # dane treningowe
│       └── keypoint_classifier.py            # logika klasyfikacji dwóch rąk
└── utils/
    └── cvfpscalc.py                # pomiar FPS
```

## Uruchomienie

### Podgląd gestów bez Unity

```bash
python app.py
```

Opcje:

| Opcja                        | Domyślnie                                                 | Opis                                                                                 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `--device`                   | `0`                                                       | numer kamery                                                                         |
| `--width`, `--height`        | `960`, `540`                                              | rozdzielczość przechwytywania                                                        |
| `--num_hands`                | `2`                                                       | liczba wykrywanych dłoni                                                             |
| `--model_path`               | `model/hand_landmarker.task`                              | model wykrywania dłoni                                                               |
| `--classifier_path`          | `model/keypoint_classifier/keypoint_classifier.tflite`    | klasyfikator (`.tflite` lub `.keras`)                                                |
| `--labels_path`              | `model/keypoint_classifier/keypoint_classifier_label.csv` | etykiety gestów                                                                      |
| `--min_detection_confidence` | `0.7`                                                     | próg wykrywania dłoni                                                                |
| `--min_tracking_confidence`  | `0.5`                                                     | próg śledzenia dłoni                                                                 |
| `--height_diff_threshold`    | `30`                                                      | minimalna różnica pionowa (px) między dłońmi; mniejsza = `inactive`, `0` = wyłączone |
| `--no_classification`        | -                                                         | tylko wykrywanie dłoni i zapis próbek, bez klasyfikatora                             |

### Wysyłanie gestów do Unity

1. Uruchom w Unity serwer TCP nasłuchujący na wybranym porcie (domyślnie `127.0.0.1:5555`).
2. Uruchom pipeline:

```bash
python integrations/gesture_pipeline.py --host 127.0.0.1 --port 5555
```

Ścieżki względne (np. `--model_path model/hand_landmarker.task`) są liczone względem bieżącego katalogu, a jeśli tam ich nie ma - względem katalogu głównego projektu, więc skrypt działa niezależnie od miejsca uruchomienia.

Dodatkowe opcje pipeline'u:

| Opcja                      | Domyślnie           | Opis                                                 |
| -------------------------- | ------------------- | ---------------------------------------------------- |
| `--host`, `--port`         | `127.0.0.1`, `5555` | adres serwera TCP w Unity                            |
| `--reconnect_delay`        | `2.0`               | przerwa (s) przed ponowną próbą połączenia           |
| `--min_gesture_confidence` | `0.0`               | minimalna pewność, aby wysłać gest                   |
| `--no_display`             | -                   | bez okna podglądu OpenCV                             |
| `--no_flip`                | -                   | bez lustrzanego odbicia obrazu                       |
| `--max_frames`             | `0`                 | zakończ po N klatkach (`0` = bez limitu)             |
| `--dry_run`                | -                   | bez łączenia z Unity, zdarzenia wypisywane w konsoli |
| `--verbose`                | -                   | dodatkowe logi                                       |

Pozostałe opcje (`--device`, `--width`, `--height`, `--model_path`, `--classifier_path`, `--labels_path`, `--num_hands`, progi pewności) działają tak samo jak w `app.py`. Zamknięcie okna podglądu: klawisz **ESC**.

### Uruchomienie z notatnika

Otwórz `integrations/run_pipeline.ipynb`, ustaw parametry w pierwszej komórce (host, port, kamera, progi, flagi `DRY_RUN` / `NO_DISPLAY` itd.) i uruchom komórki od góry do dołu. Notatnik sam wykrywa katalog główny projektu i wywołuje `integrations/gesture_pipeline.py` z tymi parametrami.

### Test bez Unity

W jednym terminalu uruchom serwer testowy, który wypisuje odebrane wiadomości:

```bash
python integrations/test_server.py
```

W drugim uruchom pipeline (z kamerą) lub wyślij pojedyncze zdarzenie:

```bash
python integrations/gesture_pipeline.py
python integrations/send_event.py gesture_started --gesture Push --confidence 0.92
```

Aby sprawdzić samo rozpoznawanie, bez żadnego serwera, użyj `--dry_run`.

## Protokół TCP

Każde zdarzenie to **jedna linia JSON zakończona znakiem `\n`** (kodowanie UTF-8). Klient łączy się z serwerem w Unity i w razie zerwania połączenia próbuje je odnowić.

Początek gestu:

```json
{
  "event": "gesture_started",
  "timestamp": 1731612345.12,
  "gesture": "Push",
  "confidence": 0.94,
  "hand_sign_id": 0
}
```

Koniec gestu:

```json
{
  "event": "gesture_ended",
  "timestamp": 1731612346.02,
  "gesture": "Push",
  "duration": 0.9,
  "confidence": 0.94
}
```

Zdarzenia przychodzą parami: `gesture_started` otwiera gest, a `gesture_ended` zamyka go (także przy zmianie gestu na inny albo przy zamknięciu pipeline'u). Klasa `inactive` nigdy nie jest wysyłana. Zdarzenia po stronie Unity warto odczytywać liniami (`ReadLine`) i parsować np. przez `JsonUtility`.

## Dostępne gesty

Etykiety znajdują się w `model/keypoint_classifier/keypoint_classifier_label.csv` (kolejność odpowiada identyfikatorom klas w modelu):

| ID  | Gest        |
| --- | ----------- |
| 0   | `Push`      |
| 1   | `Heal`      |
| 2   | `Choke`     |
| 3   | `Lightning` |
| 4   | `inactive`  |

## Dodawanie własnych gestów i trening

Dane treningowe trafiają do `model/keypoint_classifier/keypoint.csv`. Pierwsza kolumna to numer klasy, a pozostałe to 84 znormalizowane współrzędne obu dłoni.

1. Uruchom `python app.py` i klawiszem **k** włącz tryb zapisu punktów.
2. Pokaż gest obiema dłońmi i naciskaj klawisz z numerem klasy (**0-9**), aby dopisywać próbki.
3. Dopisz nazwę gestu do `keypoint_classifier_label.csv` (w tej samej kolejności co ID; `inactive` powinien być osobną klasą - służy do wyciszania ruchów niebędących gestami).
4. Wytrenuj model w notatniku treningowym (zmień `NUM_CLASSES` na aktualną liczbę klas) i zapisz wynik jako `.tflite` / `.keras` w `model/keypoint_classifier/`.

## Rozwiązywanie problemów

- **`Nie znaleziono pliku (...)`** - sprawdź ścieżki do modelu, klasyfikatora i etykiet; komunikat podaje pełną ścieżkę, której szukano.
- **`Unity niedostępne`** - serwer TCP nie nasłuchuje na podanym hoście i porcie. Pipeline działa dalej i ponawia połączenie przy wysyłaniu gestów. Do testów użyj `integrations/test_server.py`.
- **Gesty ciągle są `inactive`** - zmniejsz `--height_diff_threshold` (albo ustaw `0`) i upewnij się, że obie dłonie są widoczne w kadrze.
- **Nie można otworzyć kamery** - zmień `--device` (np. na `1`).

## Licencja

Projekt jest dostępny na licencji **Apache License 2.0**, tak jak oryginalny projekt - szczegóły w pliku [LICENSE](LICENSE).

## Autorzy i podziękowania

- Oryginalny projekt: [Kazuhito Takahashi](https://twitter.com/KzhtTkhs) - [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- Tłumaczenie na angielski i ulepszenia: [Nikita Kiselov](https://github.com/kinivi)
- [MediaPipe](https://mediapipe.dev/)

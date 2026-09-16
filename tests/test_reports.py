import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reports as rp  # noqa: E402


@pytest.mark.parametrize("text,lang", [
    ("There is a small joint effusion and the anterior cruciate ligament is intact.", "en"),
    ("Pequeño derrame articular. Ligamento cruzado anterior sin alteraciones de la señal.", "es"),
    ("Kleiner Gelenkerguss. Das vordere Kreuzband ist intakt, keine Meniskusläsion und kein Ödem.", "de"),
    ("Geen gewrichtsvocht. De voorste kruisband is intact met een normaal verloop van het signaal.", "nl"),
    ("Petit épanchement articulaire, ligament croisé antérieur sans anomalie avec des ménisques normaux.", "fr"),
    ("Mali izljev u zglobu koljena, prednja ukrižena sveza je bez promjena, menisci su očuvani.", "hr"),
    ("Diz ekleminde minimal sıvı izlendi. Ön çapraz bağ doğal görünümdedir.", "tr"),
    ("Μικρή ενδαρθρική συλλογή υγρού. Ο πρόσθιος χιαστός σύνδεσμος είναι ακέραιος.", "el"),
    ("Минимален ставен излив. Предната кръстна връзка е със запазена цялост.", "bg"),
    ("12345 ---", "unknown"),
])
def test_detect_language(text, lang):
    assert rp.detect_language(text) == lang

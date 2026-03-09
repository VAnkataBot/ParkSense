from pathlib import Path
import shutil
from ultralytics import YOLOE

model = YOLOE("yoloe-26s-seg.pt")

names = [
    # Main signs
    "blue square swedish parking allowed sign with white letter P",
    "round red circle swedish no parking prohibited sign",
    "round red circle swedish area-wide parking prohibition zone sign",
    "round red circle with diagonal cross swedish end of parking prohibition zone sign",
    # Additional plates (tilläggsskyltar)
    "rectangular white plate with hours and days text swedish time restricted parking sign",
    "rectangular white plate with coin or fee text swedish paid parking sign",
    "rectangular white plate with permit text swedish special permit required parking sign",
    "rectangular white plate with visitor or customer text swedish visitors only parking sign",
    "rectangular white plate with parking disc clock swedish parking disc required sign",
    "rectangular white plate with tenant or boende text swedish residents only parking sign",
    # Symbol plates
    "square blue plate with international wheelchair accessibility symbol white icon",
    "rectangular white plate with heavy truck lorry vehicle silhouette icon",
    "rectangular white plate with passenger car sedan vehicle silhouette icon",
    "rectangular white plate with motorcycle moped vehicle silhouette icon",
    "rectangular white plate with trailer caravan vehicle silhouette icon",
    "rectangular white plate with electric vehicle charging plug lightning bolt icon",
    # Boundary/direction plates
    "rectangular white plate with directional arrow indicating parking zone boundary",
]  # Class names for the model

model.set_classes(names, model.get_text_pe(names))

export_path = Path(model.export(format="pt", imgsz=640, simplify=True))

dest = Path("models") / "yoloe-26s-seg.pt"
dest.parent.mkdir(exist_ok=True)
shutil.move(str(export_path), dest)
print(f"Model saved to: {dest}")
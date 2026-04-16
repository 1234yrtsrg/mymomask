from pathlib import Path
from types import SimpleNamespace

from data.t2m_dataset import Text2BlendshapeDataset


def main():
    samples_dir = (Path(__file__).resolve().parent / "../samples").resolve()
    opt = SimpleNamespace(
        motion_dir=str(samples_dir),
        text_dir=str(samples_dir),
        min_motion_length=1,
        max_motion_length=0,
        unit_length=1,
        pad_to_max_length=False,
        random_crop=False,
    )

    dataset = Text2BlendshapeDataset(opt)
    caption, motion, m_length = da！taset[0]

    print(f"caption: {caption}")
    print(f"motion.shape: {tuple(motion.shape)}")
    print(f"m_length: {m_length}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest

from salat_vision.data.splits import inner_train_val, make_cv_splits
from salat_vision.data.voc import filename_series, parse_voc
from salat_vision.labels import PHASE_TO_VISUAL, PHASES, VISUAL_CLASSES, normalize_label

from .conftest import requires_data


@pytest.mark.parametrize("raw,expected", [("sitting", "julus"), ("julus", "julus"), ("Sujud", "sujud"), ("sujud ", "sujud"),
                                          ("qiyam", "qiyam"), ("ruku", "ruku"), ("dancing", None)])
def test_label_normalisation(raw, expected):
    assert normalize_label(raw) == expected


def test_every_phase_maps_to_a_visual_class():
    assert set(PHASE_TO_VISUAL) == set(PHASES)
    assert set(PHASE_TO_VISUAL.values()) == set(VISUAL_CLASSES)


def test_parse_voc_bndbox_and_polygon(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("""<annotation><filename>a.jpg</filename><path>C:\\\\users\\\\me\\\\vids\\\\a.jpg</path>
      <size><width>100</width><height>80</height><depth>3</depth></size>
      <object><name>Sujud</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>50</xmax><ymax>40</ymax></bndbox></object>
      <object><name>julus</name><polygon><x1>60</x1><y1>70</y1><x2>10</x2><y2>70</y2><x3>10</x3><y3>5</y3><x4>60</x4><y4>5</y4></polygon></object>
    </annotation>""")
    ann = parse_voc(xml)
    assert (ann.width, ann.height) == (100, 80)
    assert [o.label for o in ann.objects] == ["sujud", "julus"]
    poly = ann.objects[1]
    assert (poly.xmin, poly.ymin, poly.xmax, poly.ymax, poly.box_format) == (10, 5, 60, 70, "polygon")


@pytest.mark.parametrize("stem,expected", [("sujud31", ("sujud", 31)), ("qiam(178)", ("qiam", 178)),
                                           ("SittingMo58", ("sittingmo", 58)), ("ck2or3u694ynn", ("ck2or3u694ynn", None))])
def test_filename_series(stem, expected):
    assert filename_series(stem) == expected


def test_group_kfold_never_leaks():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 2)
    labels = np.repeat(rng.integers(0, 4, 20), 2)
    splits = make_cv_splits(labels, groups, n_splits=5, n_repeats=2, seed=1)
    assert len(splits) == 10
    for s in splits:
        assert not set(groups[s["train"]]) & set(groups[s["test"]])
        assert sorted(s["train"] + s["test"]) == list(range(40))
    tr, va = inner_train_val(np.array(splits[0]["train"]), labels, groups, val_fraction=0.2, seed=0)
    assert not set(groups[tr]) & set(groups[va])


@requires_data
def test_dataset_index_matches_audit(cfg):
    from salat_vision.data.dataset import build_index
    from salat_vision.data.splits import assign_groups

    idx = build_index(cfg=cfg)
    assert len(idx.images) == 41
    li = idx.labeled_instances
    assert len(li) == 40
    assert li["label"].value_counts().to_dict() == {"qiyam": 12, "sujud": 12, "julus": 10, "ruku": 6}
    assert len(idx.orphan_annotations) == 2
    groups, links = assign_groups(idx.images, cfg)
    li = li.assign(group_id=li["image_id"].map(groups))
    assert li["group_id"].nunique() == 29
    # the duplicated frame and the black-shirt video frames must share groups
    assert groups["ck2qcfa2rkho8083878ojzlm5"] == groups["qiam(24)"]
    assert groups["SittingMo58"] == groups["ck2or3u694ynn0757ptzgbv8t"]
    assert isinstance(links, pd.DataFrame) and len(links) > 0

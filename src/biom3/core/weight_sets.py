"""Named weight-set bundles.

A weight set clumps the Stage 1/2/3 trained checkpoints that belong together
into one JSON file (e.g. ``configs/weights/run1_base.json``) so a whole model
stack can be selected with a single ``--weight_set`` flag instead of three
separate path arguments.

Bundle paths follow the same convention as the rest of the repo's configs:
relative paths resolve against the working directory (run from the repo root),
absolute paths are used as-is. The bundle itself supports config composition
via ``load_json_config``.

A bundle can also record how its weights were trained, so a run can be checked
against it: ``pencl_trained_with_text_attention_mask`` and
``proteoscribe_trained_with_normalized_zc``, each true or false.
"""

from __future__ import annotations

import os

from biom3.core.helpers import load_json_config

WEIGHT_KEYS = ("pencl_weights", "facilitator_weights", "proteoscribe_weights")
PENCL_MASK_KEY = "pencl_trained_with_text_attention_mask"
NORMALIZED_ZC_KEY = "proteoscribe_trained_with_normalized_zc"


def load_weight_set(path):
    """Load a weight-set bundle and return the recognized weight keys.

    Unknown keys in the bundle are ignored; missing keys come back absent.
    """
    cfg = load_json_config(path)
    return {k: cfg[k] for k in WEIGHT_KEYS if k in cfg and cfg[k] is not None}


def _given_on_command_line(argv, option):
    """Whether ``option`` is on the raw command line.

    Counts argparse's ``--option=value`` form and its unambiguous abbreviations.
    """
    for token in argv or ():
        name = token.split("=", 1)[0]
        if name.startswith("--") and len(name) > 2 and option.startswith(name):
            return True
    return False


def merge_weight_set(args, weight_set_path, keys, rename=None, argv=None):
    """Fill ``args.<key>`` from a bundle for each key the run has not set.

    ``keys`` restricts which weight keys this consumer cares about (e.g. the
    embedding pipeline only needs pencl + facilitator). ``rename`` maps a bundle
    key to the attribute it fills, for consumers whose argument names differ
    (e.g. ``{"pencl_weights": "stage1_weights"}``).

    A path given on the command line always wins over the bundle. For consumers
    that also read a JSON config, pass ``argv``, the raw command line: when it
    names ``--weight_set``, the bundle in turn replaces a path that came from
    the config, because the command line outranks the config. Without ``argv``
    the bundle only fills what is unset.

    Returns ``[(attr, replaced, new)]`` for each path the bundle replaced.
    """
    if not weight_set_path or str(weight_set_path) == "None":
        return []
    rename = rename or {}
    bundle = load_weight_set(weight_set_path)
    set_on_command_line = _given_on_command_line(argv, "--weight_set")
    replaced = []
    for key in keys:
        attr = rename.get(key, key)
        current = getattr(args, attr, None)
        if not bundle.get(key):
            continue
        if current in (None, "None"):
            setattr(args, attr, bundle[key])
        elif (set_on_command_line
              and not _given_on_command_line(argv, "--" + attr)
              and not _same_file(current, bundle[key])):
            replaced.append((attr, current, bundle[key]))
            setattr(args, attr, bundle[key])
    return replaced


def _record(weight_set_path, record_key, weights_key):
    """A bundle's true/false record and the weight file it is about.

    Returns ``(None, None)`` when there is no bundle or no such record.
    """
    if not weight_set_path or str(weight_set_path) == "None":
        return None, None
    cfg = load_json_config(weight_set_path)
    value = cfg.get(record_key)
    if value is None:
        return None, None
    if not isinstance(value, bool):
        raise ValueError(
            f"{record_key} in {weight_set_path} must be true or false, "
            f"got {value!r}"
        )
    return value, cfg.get(weights_key)


def _same_file(path, other):
    return (path is not None and other is not None
            and os.path.normpath(path) == os.path.normpath(other))


def _recorded(weight_set_path, record_key, weights_key, weights_path):
    """A bundle's true/false record about one of its weight files.

    Returns None when nothing can be said: no bundle, no such record, or
    ``weights_path`` is not the file the bundle names under ``weights_key``.
    """
    value, recorded_for = _record(weight_set_path, record_key, weights_key)
    if weights_path is not None and not _same_file(weights_path, recorded_for):
        return None
    return value


def _other_file_note(weight_set_path, record_key, weights_key, weights_path):
    """Say so when a bundle's record exists but is about a different weight file."""
    value, recorded_for = _record(weight_set_path, record_key, weights_key)
    if value is None or weights_path is None or _same_file(weights_path, recorded_for):
        return None
    return (f"the weight set's record is for {recorded_for}, not the weights in use "
            f"({weights_path}), so it is not checked")


def pencl_trained_with_mask(weight_set_path, pencl_weights=None):
    """Whether a bundle records PenCL as trained with the caption attention mask.

    Returns the bundle's ``pencl_trained_with_text_attention_mask`` value, or
    None when nothing can be said: no bundle, no such key, or ``pencl_weights``
    is not the PenCL file the bundle names.
    """
    return _recorded(weight_set_path, PENCL_MASK_KEY, "pencl_weights", pencl_weights)


def proteoscribe_trained_with_normalized_zc(weight_set_path, proteoscribe_weights=None):
    """Whether a bundle records ProteoScribe as trained on unit-length z_c.

    Returns the bundle's ``proteoscribe_trained_with_normalized_zc`` value, or
    None when nothing can be said: no bundle, no such key, or
    ``proteoscribe_weights`` is not the ProteoScribe file the bundle names.
    """
    return _recorded(weight_set_path, NORMALIZED_ZC_KEY, "proteoscribe_weights",
                     proteoscribe_weights)


def check_text_attention_mask(weight_set_path, pencl_weights, text_attention_mask):
    """Compare a run's caption attention mask setting with a bundle's record.

    Returns ``(summary, warning)``: one line saying what the run does and what
    the bundle records, and the lines of a loud warning, which is empty unless
    the two disagree. The run's setting is not changed either way.
    """
    trained_with_mask = pencl_trained_with_mask(weight_set_path, pencl_weights)
    summary = "Caption attention mask: %s; %s" % (
        "on" if text_attention_mask else "off",
        _other_file_note(weight_set_path, PENCL_MASK_KEY, "pencl_weights", pencl_weights)
        or {True: "the weight set records PenCL as trained with it",
            False: "the weight set records PenCL as trained without it",
            None: "how PenCL was trained is not recorded"}[trained_with_mask],
    )
    if trained_with_mask is None or trained_with_mask == bool(text_attention_mask):
        return summary, []
    bar = "!" * 72
    return summary, [
        bar,
        "CAPTION ATTENTION MASK DOES NOT MATCH THE WEIGHTS",
        f"{weight_set_path} records PenCL as trained "
        f"{'with' if trained_with_mask else 'without'} the mask "
        f"({PENCL_MASK_KEY}),",
        f"but this run has --text_attention_mask "
        f"{'on' if text_attention_mask else 'off'}.",
        "z_t will not be what these weights were trained to produce.",
        bar,
    ]


def check_normalize_zc(weight_set_path, proteoscribe_weights, normalize_zc):
    """Compare a run's normalize_zc setting with a bundle's record.

    Returns ``(summary, warning)`` as :func:`check_text_attention_mask` does:
    the warning is empty unless the two disagree, and the run's setting is not
    changed either way.
    """
    trained_normalized = proteoscribe_trained_with_normalized_zc(
        weight_set_path, proteoscribe_weights)
    summary = "Conditioning vector scaled to unit length: %s; %s" % (
        "yes" if normalize_zc else "no",
        _other_file_note(weight_set_path, NORMALIZED_ZC_KEY, "proteoscribe_weights",
                         proteoscribe_weights)
        or {True: "the weight set records ProteoScribe as trained that way",
            False: "the weight set records ProteoScribe as trained without it",
            None: "how ProteoScribe was trained is not recorded"}[trained_normalized],
    )
    if trained_normalized is None or trained_normalized == bool(normalize_zc):
        return summary, []
    bar = "!" * 72
    return summary, [
        bar,
        "Z_C NORMALISATION DOES NOT MATCH THE WEIGHTS",
        f"{weight_set_path} records ProteoScribe as trained "
        f"{'with' if trained_normalized else 'without'} unit-length conditioning "
        f"vectors ({NORMALIZED_ZC_KEY}),",
        f"but this run has --normalize_zc {'on' if normalize_zc else 'off'}.",
        "The model will be conditioned on vectors of a length it was not trained on.",
        bar,
    ]

#! /usr/bin/env python
#
# Tests for textual inversion (TI).
#
# note:
# - old-style logging % formatting is unreadable
#   pylint: disable=logging-fstring-interpolation
#

"""Tests for train_textual_inversion.py"""

# Standard modules
import argparse
import glob
import logging
import os
import re

# Installed modules
from PIL import Image
from clip_interrogator import Config, Interrogator
from mezcla import debug

# Local modules
from library import train_util
from library.utils import setup_logging
from train_textual_inversion import setup_parser, TextualInversionTrainer
from tests.nearest_tokens_for_embedding import load_textual_inversion_embedding, find_closest_tokens

# Constants
# note: Optionally enable use of known good TI embedding models and generated samples
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL")
MOCK_TESTS = (os.getenv("MOCK_TESTS") == "1")
MOCK_TI_EMBEDDING_PATH = os.getenv("MOCK_TI_EMBEDDING_PATH") or (MOCK_TESTS and "tests/mock-data/ti-impressionism-sd1-4.safetensors")
MOCK_IMAGE_SAMPLE_SPEC = os.getenv("MOCK_IMAGE_SAMPLE_SPEC") or (MOCK_TESTS and "tests/mock-data/*.png")
CONFIG_FILE = os.getenv("CONFIG_FILE") or os.path.join("tests", "TI-AdamW8bit.toml")


def has_image_extension(file_path):
    """Whether FILE_PATH ends in supported image"""
    debug.assertion(all(ext.startswith for ext in train_util.IMAGE_EXTENSIONS))
    ext_regex = "|".join(train_util.IMAGE_EXTENSIONS)
    ok = re.search(fr"({ext_regex})$", file_path)    
    logging.debug(f"has_image_extension({file_path}) => {ok}")
    return ok


def image_to_text(path):
    """Generate text from image at PATH"""
    image = Image.open(path).convert('RGB')
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    description = ci.interrogate(image)
    debug.trace_expr(5, path, description)
    return description


def init(pytestconfig=None, config_file=None):
    """Initialize for testing sd-scripts using options from CONFIG_FILE.
    note: Optionally uses command line arguments via pytest custom arg (see conftest.py)."""
    if config_file is None:
        config_file = CONFIG_FILE

    # Initialize args from configuration file
    init_args = argparse.Namespace()
    init_args.output_config = False
    init_args.config_file = config_file
    debug.trace_expr(6, init_args)
    parser = setup_parser()
    args = train_util.read_config_from_file(init_args, parser)
    debug.trace_expr(7, args)

    # Override from command line (via pytest --custom_args)
    # note: This is just intended for adhoc testing
    if pytestconfig:
        custom_args = pytestconfig.getoption("--custom_args").split()
        if custom_args:
            default_args = parser.parse_args([])
            command_line_args = parser.parse_args(custom_args)
            debug.trace_expr(5, command_line_args)
            for key, value in vars(command_line_args).items():
                if value != getattr(default_args, key):
                    setattr(args, key, value)
    debug.trace_expr(6, args)

    # Setup for training
    # note: uses process-specific subdir for output (to avoid reading old images)
    args.output_dir = os.path.join(args.output_dir, str(os.getpid()))
    os.makedirs(args.output_dir, exist_ok=True)
    log_level = LOGGING_LEVEL
    setup_logging(args, log_level=log_level, reset=True)
    logging.info(f"init: {logging.root.level=}")

    # Train the inversion
    if not MOCK_TI_EMBEDDING_PATH:
        trainer = TextualInversionTrainer()
        trainer.train(args)
        debug.trace_object(5, trainer)

    return args


def test_train_simple_ti_img2txt(pytestconfig):
    """Verify simple textual inversion via img2txt"""
    debug.trace(5, f"test_train_simple_ti_img2txt({pytestconfig})")

    # Load configuration and train model
    args = init(pytestconfig=pytestconfig)
    
    # Make sure at lease one of sample images reflect inversion
    # note: this uses clip-style caption generation to verify
    sample_dir = os.path.join(args.output_dir, "sample")
    num_ok = 0
    image_spec = (MOCK_IMAGE_SAMPLE_SPEC or os.path.join(sample_dir, "*.*"))
    sample_images = glob.glob(image_spec)
    for img in sample_images:
        if not has_image_extension(img):
            logging.debug(f"FYI: skipping non-image file {img!r}")
            continue
        description = image_to_text(img)
        if re.search(r"\b(painting|style)\b", description, flags=re.IGNORECASE):
            num_ok += 1
        else:
            logging.debug(f"FYI: problem with description for {img!r}")
    ok_threshold = len(sample_images) // 2
    debug.trace_expr(5, ok_threshold, num_ok)
    assert num_ok >= ok_threshold


def test_train_simple_ti_embedding_proximity(pytestconfig):
    """Verify simple textual inversion via embedding token proximity"""
    debug.trace(5, f"test_train_simple_ti_embedding_proximity({pytestconfig})")
    
    # Load configuration and train model
    args = init(pytestconfig=pytestconfig)
    
    # Make sure at least one token among those known to be related to concept in closest tokens
    ti_path = (MOCK_TI_EMBEDDING_PATH or os.path.join(args.output_dir, f"{args.output_name}.safetensors"))
    pipe, learned_embeds = load_textual_inversion_embedding(
        args.pretrained_model_name_or_path, ti_path)
    expected_close_tokens = ["monet", "impressionism", "impressionist"]
    nearest_close_tokens = find_closest_tokens(pipe, learned_embeds, 100)
    num_ok = 0
    for token in expected_close_tokens:
        token_regex = f"{token}(</w>)?"
        if any(re.search(token_regex, t) for (t, _score) in nearest_close_tokens):
            num_ok += 1
        else:
            logging.debug(f"FYI: problem verifying closeness for {token!r}")
    assert num_ok > 0

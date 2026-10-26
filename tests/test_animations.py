"""Global: animations preserve scientific time and scale; local: fixtures never enter public reports.

Sources: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif.
"""
import numpy as np  # Local: small independent known frame sequences.
from PIL import Image  # Local: decode the actual GIF, not just renderer metadata.
from gphyt_attnres_study.animations import render_gif  # Global: exercise the encoder used for publication.


def test_gif_keeps_frames_timing_and_all_time_color_limits(tmp_path):
    """Global: dynamic autoscaling could conceal instability; local: the last frame sets an extreme."""
    panels = np.zeros((2, 3, 4, 2))  # Local: two panels, three forecast steps, four by two grid points.
    panels[0, :, 0, 0] = [0, 1, 2]  # Local: retain a visibly evolving ground-truth pixel.
    panels[1, :, 0, 0] = [-2, 1, 8]  # Local: late forecast amplitude must set the shared range from the start.
    panels[1, 1, 1, 1] = np.nan  # Global: missing predictions remain marked, not repaired.
    path = tmp_path / 'comparison.gif'  # Local: isolated fixture artifact.
    result = render_gif(panels, ['Truth', 'Prediction'], path, 'Test field', 'Test group')  # Local: use the real standard plotting/encoding path.
    assert result['vmin'] == -2 and result['vmax'] == 8 and result['frames'] == 3  # Global: one range spans every panel and time.
    with Image.open(path) as animation:
        assert animation.n_frames == 3 and animation.info['loop'] == 0  # Local: an animated, repeating output rather than a static first frame.
        frames = []  # Local: inspect decoded images for time evolution and explicit divergence color.
        for step in range(animation.n_frames):
            animation.seek(step)  # Local: decode each stored frame in sequence.
            assert animation.info['duration'] == 200  # Global: five forecast frames per second, independent of physical time units.
            frames.append(np.asarray(animation.convert('RGB')))  # Local: compare pixels after actual GIF quantization.
        assert not np.array_equal(frames[0], frames[-1])  # Global: the published file visibly evolves.
        assert ((frames[1][..., 0] > 220) & (frames[1][..., 1] < 40) & (frames[1][..., 2] > 220)).any()  # Global: nonfinite forecast pixels remain visibly magenta after encoding.

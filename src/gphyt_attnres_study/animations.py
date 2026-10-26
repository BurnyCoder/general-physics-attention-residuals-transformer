"""Global: animate measured autoregressive forecasts without additional model execution.

Local: reuse matched-case validation, Matplotlib's fixed normalization and Pillow GIF writer.
Sources: https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html and
https://matplotlib.org/stable/gallery/images_contours_and_fields/multi_image.html.
"""
import matplotlib  # Local: headless CPU rendering works in the same locked training environment.
matplotlib.use('Agg')  # Local: no native plotting windows or GPU contexts are created.
import matplotlib.pyplot as plt  # Local: ordinary scientific plots retain labels and an actual colorbar.
from matplotlib.animation import PillowWriter  # Local: the installed Matplotlib/Pillow stack encodes genuine GIF frames.
from matplotlib.colors import Normalize  # Global: identical values share one color across every panel and time.
import numpy as np  # Local: preserve measured values until display quantization.
from .comparisons import matched_fields, MODES  # Global: still figures and animations use the same full-target alignment check.
from .runtime import PROJECT, LOG, digest, root, stamp, write_json  # Local: provenance identifies exact input arrays and output files.

TITLES = ('GPhyT without\nAttention Residuals', 'Block Attention\nResiduals', 'Full Attention\nResiduals', 'Uniform Block\n(fixed averaging)')  # Global: baseline, treatment, and controls are unmistakable in every frame.
GROUPS = {
    'trained': ('Trained models: 3,175 updates', ('persistence', *[f'{mode}_seed0' for mode in MODES]), ('Ground truth', 'Persistence', *TITLES)),  # Local: all four trained architectures and the parameter-free baseline.
    'initialized': ('Initialized models: 0 updates (post-hoc)', ('persistence', *[f'untrained_{mode}_seed0' for mode in MODES]), ('Ground truth', 'Persistence', *TITLES)),  # Global: zero-update models are visibly distinguished from trained outputs.
    'interventions': ('Routing interventions: same trained Block checkpoint', ('persistence', 'standard_seed0', 'block_seed0', 'block_seed0_uniform', 'block_seed0_shuffle'), ('Ground truth', 'Persistence', TITLES[0], 'Block: original\nlearned routing', 'Block: uniform\nrouting intervention', 'Block: shuffled\n(random) routing')),  # Global: random routing is a within-model intervention, not another trained architecture.
}  # Local: group membership is fixed by the previously published comparisons, never selected by appearance.
CASES = (('turbulent_radiative_layer_2D', 1, 'Radiative flow / density', 'radiative_flow'), ('rayleigh_benard_uniform', 2, 'Convection / buoyancy (not temperature)', 'convection'))  # Local: reuse the existing representative fields and predetermined cases.


def render_gif(panels, titles, path, field, group):
    """Local: encode aligned [panel,time,x,y] fields; global: one unclipped scale for the whole clip."""
    finite = panels[np.isfinite(panels)]  # Local: nonfinite pixels are masked visibly rather than used as color limits.
    minimum, maximum = float(finite.min()), float(finite.max())  # Global: include all forecast extrema, not only the target or first frame.
    norm = Normalize(minimum, maximum)  # Local: this same normalization object survives every animation update.
    color = plt.get_cmap('viridis').with_extremes(bad='magenta')  # Global: instability cannot silently become a plausible field value.
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout='constrained')  # Local: six equal panels remain legible in a GitHub README.
    images = []  # Local: update existing image artists rather than redraw their axes and scales.
    for index, axis in enumerate(axes.flat):
        if index >= len(panels):
            axis.set_axis_off()  # Local: small isolated test cases do not create invented extra model panels.
            continue  # Local: only actual supplied fields are rendered.
        image = axis.imshow(np.ma.masked_invalid(panels[index, 0].T), origin='lower', interpolation='nearest', norm=norm, cmap=color)  # Global: transpose [x,y] to screen coordinates without smoothing or spatial interpolation.
        axis.set(title=titles[index], xlabel='x grid index', ylabel='y grid index', xticks=[0, panels.shape[2]-1], yticks=[0, panels.shape[3]-1])  # Local: explicit grid indices avoid invented physical distances.
        axis.title.set_fontsize(12)  # Local: model names remain readable at native GIF resolution.
        images.append(image)  # Local: artist order equals source panel order for every time step.
    bar = figure.colorbar(images[0], ax=axes, location='bottom', fraction=.05, pad=.06)  # Global: one shared colorbar states the numerical meaning of every panel.
    bar.set_label('Training-standardized value / fixed across panels and time / magenta = nonfinite')  # Local: standardized negative values are not themselves physical negative densities.
    caption = f'{field} / {group}\nSeed 0 / first test trajectory and window / forecast step '  # Local: a constant prefix exposes the fixed case and reserves the same title space.
    heading = figure.suptitle(caption + f'01/{panels.shape[1]}', fontsize=14)  # Local: include both title lines before arranging panels.
    figure.canvas.draw()  # Local: resolve labels/colorbar once before animation; https://matplotlib.org/stable/api/layout_engine_api.html.
    figure.set_layout_engine('none')  # Global: frozen panel positions prevent layout motion from masquerading as field evolution.
    temporary = path.with_suffix('.tmp.gif')  # Local: interrupted rendering cannot leave a complete-looking final GIF.
    writer = PillowWriter(fps=5)  # Local: 200 ms per forecast step; playback speed is not a claim about physical time.
    try:
        with writer.saving(figure, str(temporary), dpi=100):
            for step in range(panels.shape[1]):
                for image, sequence in zip(images, panels):
                    image.set_data(np.ma.masked_invalid(sequence[step].T))  # Global: replay the exact saved forecast, including its failures.
                heading.set_text(caption + f'{step+1:02d}/{panels.shape[1]}')  # Local: only the saved forecast index changes inside the fixed title region.
                writer.grab_frame()  # Local: standard PillowWriter stores the fully labeled current Matplotlib canvas.
                if step == 0:
                    figure.savefig(path.with_suffix('.png'), dpi=100)  # Local: a real first-frame poster supports a nonmoving viewer and explicit playback control.
        temporary.replace(path)  # Global: publish only the complete looping GIF.
    finally:
        plt.close(figure)  # Local: each completed or failed clip releases its plotting resources.
    return dict(frames=panels.shape[1],duration_ms=200,vmin=minimum,vmax=maximum,loop=0)  # Global: encoding and scale choices accompany the file in the public provenance manifest.


def rollout_gifs():
    """Global: six fixed comparisons cover every model/control; local: CPU-only saved-array replay."""
    folder = PROJECT / 'reports/animations'  # Local: small public supplements live with the experiment reports.
    folder.mkdir(parents=True, exist_ok=True)  # Local: no raw data or private configuration are copied here.
    entries = []  # Local: incomplete groups remain explicitly unavailable.
    for dataset, channel, field, slug in CASES:
        for key, (group, runs, titles) in GROUPS.items():
            name = f'{key}_{slug}.gif'  # Local: readable stable URLs support README and Pages embeds.
            panels = matched_fields(root() / 'visuals', dataset, channel, step=None, runs=runs)  # Global: validate full targets/masks and all forecast dimensions before presentation.
            entry = dict(file=name,dataset=dataset,field=field,group=key,label=group,panels=list(titles),status='unavailable')  # Local: preserve planned clips even when measurements are absent.
            if panels is not None:
                if panels.shape[1] != 24:
                    raise ValueError(f'Expected all 24 saved forecast steps: {name}')  # Global: a truncated clip cannot silently masquerade as the complete rollout.
                entry.update(render_gif(panels, titles, folder / name, field, group),status='complete',sha256=digest(folder / name))  # Local: store the verified encoder settings and exact output hash.
                entry['sources'] = {f'{run}_{dataset}.npz': digest(root() / 'visuals' / f'{run}_{dataset}.npz') for run in runs}  # Global: every public example traces to unchanged full-precision saved arrays.
            entries.append(entry)  # Local: missing groups remain in the manifest rather than disappearing.
            LOG.info('Animation %s: %s', name, entry['status'])  # Global: timestamps identify actual progress and unavailable evidence.
    write_json(folder / 'manifest.json',dict(created=stamp(),selection='Seed 0, first test trajectory, first predetermined window, steps 1-24',entries=entries))  # Local: relative source identifiers avoid publishing local paths.
    return entries  # Local: the dashboard and website use the same generated publication inventory.

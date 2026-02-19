#!/usr/bin/env python3
"""F_register_diagram.py

Generate an SVG diagram showing non-clashing register positions
as sliding windows below the scaffold DNA sequence, colored by
min_dist using the cividis colormap.

Outputs:
  {guest_code}_register_diagram.svg

Usage:
    python F_register_diagram.py <guest_code> [scaffold_code]
    python F_register_diagram.py 1jgg 9YZJ
"""
import csv, sys, json, os

guest_code = sys.argv[1] if len(sys.argv) > 1 else '1jgg'
scaffold_code = sys.argv[2] if len(sys.argv) > 2 else '9YZJ'

outdir = f'output/{guest_code}.{scaffold_code}'
csv_path = f'{outdir}/{guest_code}_ranked.csv'
svg_path = f'{outdir}/{guest_code}_register_diagram.svg'
scaffold_json = f'scaffold_models/{scaffold_code}.json'

# ── Cividis colormap ──────────────────────────────────────────────────
CIVIDIS = [
    (0.00, (0.000, 0.135, 0.305)),
    (0.10, (0.065, 0.176, 0.353)),
    (0.20, (0.135, 0.219, 0.396)),
    (0.30, (0.210, 0.264, 0.425)),
    (0.40, (0.290, 0.312, 0.438)),
    (0.50, (0.373, 0.364, 0.435)),
    (0.60, (0.461, 0.418, 0.416)),
    (0.70, (0.558, 0.477, 0.384)),
    (0.80, (0.666, 0.541, 0.336)),
    (0.90, (0.787, 0.614, 0.269)),
    (1.00, (0.993, 0.906, 0.144)),
]

def cividis_rgb(t):
    """Interpolate cividis at t in [0,1]. Returns (r,g,b) in 0-1."""
    t = max(0.0, min(1.0, t))
    for i in range(len(CIVIDIS) - 1):
        t0, c0 = CIVIDIS[i]
        t1, c1 = CIVIDIS[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(c0[j] + f * (c1[j] - c0[j]) for j in range(3))
    return CIVIDIS[-1][1]

def cividis_hex(t):
    r, g, b = cividis_rgb(t)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def text_color(t):
    """White text on dark backgrounds, black on light."""
    r, g, b = cividis_rgb(t)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#ffffff' if lum < 0.45 else '#000000'

# ── Load scaffold info ────────────────────────────────────────────────
with open(scaffold_json) as f:
    scaff = json.load(f)

cat_json = f'{outdir}/{guest_code}_categories.json'
with open(cat_json) as f:
    cat_data = json.load(f)

top_chain = cat_data['top_chain']
dna_chains = scaff['dna_chains']
bot_chain = [c for c in dna_chains if c != top_chain][0]
top_seq = scaff['chains'][top_chain]['sequence']
WC = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
bot_display = ''.join(WC[b] for b in top_seq)
seq_len = len(top_seq)

# Scaffold protein binding footprint
foot = cat_data.get('core_footprint', {})
foot_start = foot.get('start')
foot_end = foot.get('end')

# ── Parse ranked CSV ──────────────────────────────────────────────────
registers = []
with open(csv_path) as f:
    lines = [l for l in f if not l.startswith('#')]
reader = csv.DictReader(lines)
for row in reader:
    if row['classification'] != 'CLASH':
        registers.append({
            'rank':       int(row['rank']),
            'start_pos':  int(row['start_pos']),
            'end_pos':    int(row['end_pos']),
            'orientation': row['orientation'],
            'min_dist':   float(row['min_dist']),
        })

fwd = sorted([r for r in registers if r['orientation'] == 'fwd'],
             key=lambda r: r['start_pos'])
rev = sorted([r for r in registers if r['orientation'] == 'rev'],
             key=lambda r: r['start_pos'])

print(f"  {len(fwd)} fwd + {len(rev)} rev non-clashing registers")

# ── Color mapping: min_dist -> cividis ────────────────────────────────
# Largest min_dist -> blue (t=0), smallest -> yellow (t=1)
all_dists = [r['min_dist'] for r in registers]
d_lo, d_hi = min(all_dists), max(all_dists)
d_range = d_hi - d_lo if d_hi > d_lo else 1.0

def dist_to_t(d):
    return (d_hi - d) / d_range

# ── SVG layout (all values in mm, which is the viewBox unit) ─────────
BP_W       = 1.7     # horizontal space per base pair
BAR_H      = 3.6     # bar height
BAR_GAP    = 0.6     # vertical gap between bars
MARGIN_L   = 22.0    # left margin (room for brace + rotated text + padding)
MARGIN_R   = 8.0     # right margin
SEQ_FONT   = 2.6     # font size for sequence letters
NUM_FONT   = 2.6     # font size for position numbers
LABEL_FONT = 2.0     # font size for bar labels
BRACE_FONT = 3.12    # font size for brace labels (increased 30%)
SECTION_GAP = 5.0    # gap between fwd and rev sections
SCALE_H    = 3.0     # scale bar height
SCALE_GAP  = 6.0     # gap before scale bar

# Brace geometry
BRACE_END_X = 9.5    # x of brace top/bottom ends (right side)
BRACE_TIP_X = 7.0    # x of brace midpoint tip (left side)
BRACE_LBL_X = 4.5    # x center of rotated text label

def bp_x(pos):
    """X of the center of bp position pos (1-indexed)."""
    return MARGIN_L + (pos - 1) * BP_W + BP_W / 2

def bp_left(pos):
    """X of the left edge of bp position pos."""
    return MARGIN_L + (pos - 1) * BP_W

# ── Vertical positions ────────────────────────────────────────────────
SUPER_H    = 4.0     # height of scaffold protein super-register bar
SUPER_GAP  = 2.0     # gap after super-register
y_super    = 4.0                             # top of super-register bar
y_num      = y_super + SUPER_H + SUPER_GAP + 2.0
y_top      = y_num + 4.5
y_bot      = y_top + 3.5
y_fwd0     = y_bot + 5.0             # first fwd bar
y_rev0     = y_fwd0 + len(fwd) * (BAR_H + BAR_GAP) + SECTION_GAP
y_bars_end = y_rev0 + len(rev) * (BAR_H + BAR_GAP)
y_scale    = y_bars_end + SCALE_GAP
total_h    = y_scale + SCALE_H + 8
total_w    = MARGIN_L + seq_len * BP_W + MARGIN_R

# ── Build SVG ─────────────────────────────────────────────────────────
s = []
a = s.append

a('<?xml version="1.0" encoding="UTF-8"?>')
a(f'<svg xmlns="http://www.w3.org/2000/svg" '
  f'width="{total_w:.1f}mm" height="{total_h:.1f}mm" '
  f'viewBox="0 0 {total_w:.1f} {total_h:.1f}">')

# Styles
a('<style>')
a(f'  .seq {{ font-family: Courier, monospace; font-size: {SEQ_FONT}px; '
  f'text-anchor: middle; fill: #1a1a1a; }}')
a(f'  .num {{ font-family: Arial, sans-serif; font-size: {NUM_FONT}px; '
  f'text-anchor: middle; fill: #333; }}')
a(f'  .barlbl {{ font-family: Arial, sans-serif; font-size: {LABEL_FONT}px; '
  f'text-anchor: middle; dominant-baseline: central; }}')
a(f'  .bracelbl {{ font-family: Arial, sans-serif; font-size: {BRACE_FONT}px; '
  f'fill: #444; text-anchor: middle; }}')
a(f'  .scalelbl {{ font-family: Arial, sans-serif; font-size: {NUM_FONT}px; '
  f'fill: #333; }}')
a(f'  .superlbl {{ font-family: Arial, sans-serif; font-size: {LABEL_FONT + 0.3}px; '
  f'text-anchor: middle; dominant-baseline: central; fill: #1a5276; }}')
a('</style>')

# ── Scaffold protein super-register (above sequence ruler) ───────────
if foot_start is not None and foot_end is not None:
    sx = bp_left(foot_start)
    sw = (foot_end - foot_start + 1) * BP_W
    a(f'<rect x="{sx:.2f}" y="{y_super:.2f}" width="{sw:.2f}" '
      f'height="{SUPER_H}" rx="1.2" fill="#aed6f1" stroke="#5dade2" '
      f'stroke-width="0.5"/>')
    scx = sx + sw / 2
    scy = y_super + SUPER_H / 2
    a(f'<text class="superlbl" x="{scx:.2f}" y="{scy:.2f}">'
      f'Scaffold Protein: {foot_start}\u2013{foot_end}</text>')

# ── Background column markers (every 5 bp) ───────────────────────────
col_top = y_num - NUM_FONT
col_h = y_bars_end + 1 - col_top
for p in range(1, seq_len + 1):
    if (p - 1) % 5 == 0:
        x = bp_x(p) - BP_W * 0.4
        color = '#f0f0f0' if (p - 1) % 10 == 0 else '#f5f5f5'
        a(f'<rect x="{x:.2f}" y="{col_top:.1f}" '
          f'width="{BP_W * 0.8:.2f}" height="{col_h:.1f}" '
          f'fill="{color}"/>')

# ── Position numbers ─────────────────────────────────────────────────
for p in range(1, seq_len + 1):
    if (p - 1) % 5 == 0:
        a(f'<text class="num" x="{bp_x(p):.2f}" y="{y_num}">{p}</text>')

# ── DNA sequences ─────────────────────────────────────────────────────
# Top strand 5'->3'
a(f'<text class="seq" x="{bp_x(1) - BP_W:.2f}" y="{y_top}" '
  f'style="font-size:{SEQ_FONT*0.7:.1f}px;fill:#999">5\'</text>')
for i, base in enumerate(top_seq):
    a(f'<text class="seq" x="{bp_x(i+1):.2f}" y="{y_top}">{base}</text>')
a(f'<text class="seq" x="{bp_x(seq_len) + BP_W:.2f}" y="{y_top}" '
  f'style="font-size:{SEQ_FONT*0.7:.1f}px;fill:#999">3\'</text>')

# Bottom strand 3'->5'
a(f'<text class="seq" x="{bp_x(1) - BP_W:.2f}" y="{y_bot}" '
  f'style="font-size:{SEQ_FONT*0.7:.1f}px;fill:#999">3\'</text>')
for i, base in enumerate(bot_display):
    a(f'<text class="seq" x="{bp_x(i+1):.2f}" y="{y_bot}">{base}</text>')
a(f'<text class="seq" x="{bp_x(seq_len) + BP_W:.2f}" y="{y_bot}" '
  f'style="font-size:{SEQ_FONT*0.7:.1f}px;fill:#999">5\'</text>')

# ── Curly brace helper ───────────────────────────────────────────────
def draw_brace(y_start, n_bars, label):
    """Draw a left-facing vertical curly brace with rotated label.

    Shape: curly-hook elbows at top/bottom, straight inner arms,
    smooth cubic-bezier tip at the midpoint.
    """
    if n_bars == 0:
        return
    yt = y_start
    yb = y_start + n_bars * (BAR_H + BAR_GAP) - BAR_GAP
    ym = (yt + yb) / 2
    ex = BRACE_END_X   # right side (arm endpoints)
    tx = BRACE_TIP_X   # inner vertical x
    dx = ex - tx        # horizontal span of elbows
    tip_off = 1.0       # how far tip extends left of inner arm
    tip_h = 1.5         # half-height of the tip curve

    # Clamp for very short braces
    half = (yb - yt) / 2
    if dx + tip_h > half:
        scale = half / (dx + tip_h)
        dx *= scale
        tip_h *= scale

    arm_top = ym - tip_h - (yt + dx)   # inner arm length (top half)
    arm_bot = (yb - dx) - (ym + tip_h) # inner arm length (bottom half)

    path = (
        # Top elbow with curly hook (cp1 goes left-and-up)
        f'M {ex},{yt:.2f} '
        f'C {ex - 0.818*dx:.4f},{yt - 0.052*dx:.4f} '
        f'{tx},{yt + 0.448*dx:.4f} '
        f'{tx},{yt + dx:.4f} '
        # Straight inner arm down
        f'v {arm_top:.4f} '
        # Smooth tip (two cubic bezier segments)
        f'c {-0.182*tip_off:.4f},{0.535*tip_h:.4f} '
        f'{-0.515*tip_off:.4f},{0.869*tip_h:.4f} '
        f'{-tip_off:.4f},{tip_h:.4f} '
        f'{0.561*tip_off:.4f},{0.131*tip_h:.4f} '
        f'{0.894*tip_off:.4f},{0.465*tip_h:.4f} '
        f'{tip_off:.4f},{tip_h:.4f} '
        # Straight inner arm down
        f'v {arm_bot:.4f} '
        # Bottom elbow with curly hook (cp2 stays near inner x)
        f'c 0,{0.552*dx:.4f} '
        f'{0.321*dx:.4f},{dx + 0.077*dx:.4f} '
        f'{dx:.4f},{dx:.4f}'
    )
    a(f'<path d="{path}" fill="none" stroke="#555" stroke-width="0.4"/>')
    # Rotated label
    a(f'<text class="bracelbl" '
      f'transform="translate({BRACE_LBL_X:.1f},{ym:.2f}) rotate(-90)">'
      f'{label}</text>')

# ── Register bars ─────────────────────────────────────────────────────
def overlaps_footprint(start, end):
    """Check if register [start, end] overlaps scaffold protein footprint."""
    if foot_start is None or foot_end is None:
        return False
    return start <= foot_end and end >= foot_start

def draw_bars(regs, y_start):
    for i, r in enumerate(regs):
        y = y_start + i * (BAR_H + BAR_GAP)
        x = bp_left(r['start_pos'])
        w = (r['end_pos'] - r['start_pos'] + 1) * BP_W
        # Color by min_dist
        t = dist_to_t(r['min_dist'])
        fill = cividis_hex(t)
        tc = text_color(t)
        # Magenta border if overlapping scaffold protein footprint
        overlap = overlaps_footprint(r['start_pos'], r['end_pos'])
        stroke = ' stroke="#c71585" stroke-width="0.3"' if overlap else ''
        # Single rounded rectangle spanning full window
        a(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
          f'height="{BAR_H}" rx="0.8" fill="{fill}"{stroke}/>')
        # Centered label
        txt = f"{r['start_pos']}\u2013{r['end_pos']}"
        cx = x + w / 2
        cy = y + BAR_H / 2
        a(f'<text class="barlbl" x="{cx:.2f}" y="{cy:.2f}" '
          f'fill="{tc}">{txt}</text>')

draw_brace(y_fwd0, len(fwd), 'Forward alignment')
draw_bars(fwd, y_fwd0)
draw_brace(y_rev0, len(rev), 'Reverse alignment')
draw_bars(rev, y_rev0)

# ── Color scale bar ──────────────────────────────────────────────────
scale_x = MARGIN_L
scale_w = seq_len * BP_W
n_steps = 64
step_w = scale_w / n_steps

a('<!-- color scale bar -->')
a(f'<text class="scalelbl" x="{scale_x:.1f}" y="{y_scale - 1:.1f}">'
  f'min_dist (\u00c5)</text>')

for k in range(n_steps):
    t = k / (n_steps - 1)
    sx = scale_x + k * step_w
    fill = cividis_hex(t)
    a(f'<rect x="{sx:.2f}" y="{y_scale:.1f}" '
      f'width="{step_w + 0.1:.2f}" height="{SCALE_H}" fill="{fill}"/>')

# Scale tick labels
n_ticks = 5
for k in range(n_ticks + 1):
    frac = k / n_ticks
    val = d_hi - frac * d_range  # left=d_hi (blue), right=d_lo (yellow)
    tx = scale_x + frac * scale_w
    ty = y_scale + SCALE_H + NUM_FONT + 0.5
    anchor = 'start' if k == 0 else ('end' if k == n_ticks else 'middle')
    a(f'<text class="scalelbl" x="{tx:.2f}" y="{ty:.1f}" '
      f'text-anchor="{anchor}">{val:.1f}</text>')

a('</svg>')

with open(svg_path, 'w') as f:
    f.write('\n'.join(s))

print(f"  SVG -> {svg_path}")
print(f"  min_dist range: {d_lo:.1f} - {d_hi:.1f} A")

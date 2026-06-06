def build_svg(inputs, outputs, fname, title):
    W, H = 820, 1080
    in_x, hid_x, out_x = 130, 410, 690
    hid_top, hid_bot = 90, 1010
    n_hid = 128
    hid_r = 2.8
    node_r = 18

    # hidden y positions
    hid_ys = [hid_top + (hid_bot - hid_top) * i / (n_hid - 1) for i in range(n_hid)]

    # center the input/output nodes vertically
    def col_ys(n, spacing=82):
        mid = (hid_top + hid_bot) / 2
        total = (n - 1) * spacing
        start = mid - total / 2
        return [start + i * spacing for i in range(n)]

    in_ys = col_ys(len(inputs))
    out_ys = col_ys(len(outputs))

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # ---- connections: input -> hidden ----
    parts.append('<g stroke="#7F77DD" stroke-width="0.4" opacity="0.12">')
    for iy in in_ys:
        for hy in hid_ys:
            parts.append(f'e x1="{in_x}" y1="{iy:.1f}" x2="{hid_x}" y2="{hy:.1f}"/>')
    parts.append('</g>')

    # ---- connections: hidden -> output ----
    parts.append('<g stroke="#1D9E75" stroke-width="0.4" opacity="0.11">')
    for hy in hid_ys:
        for oy in out_ys:
            parts.append(f'e x1="{hid_x}" y1="{hy:.1f}" x2="{out_x}" y2="{oy:.1f}"/>')
    parts.append('</g>')

    # ---- hidden neurons (all 128) ----
    parts.append('<g fill="#EEEDFE" stroke="#534AB7" stroke-width="0.6">')
    for hy in hid_ys:
        parts.append(f'ircle cx="{hid_x}" cy="{hy:.1f}" r="{hid_r}"/>')
    parts.append('</g>')

    # ---- input nodes ----
    for iy, lab in zip(in_ys, inputs):
        parts.append(
            f'ircle cx="{in_x}" cy="{iy:.1f}" r="{node_r}" '
            f'fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{in_x-node_r-10}" y="{iy+4:.1f}" text-anchor="end" '
            f'font-size="15" fill="#042C53">{lab}</text>'
        )

    # ---- output nodes ----
    for oy, lab in zip(out_ys, outputs):
        parts.append(
            f'ircle cx="{out_x}" cy="{oy:.1f}" r="{node_r}" '
            f'fill="#E1F5EE" stroke="#0F6E56" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{out_x}" y="{oy+4:.1f}" text-anchor="middle" '
            f'font-size="14" fill="#04342C">{lab}</text>'
        )
        parts.append(
            f'<text x="{out_x+node_r+10}" y="{oy+4:.1f}" text-anchor="start" '
            f'font-size="13" fill="#5F5E5A">output</text>'
        )

    # ---- layer headers ----
    parts.append(
        f'<text x="{in_x}" y="45" text-anchor="middle" '
        f'font-size="17" font-weight="bold" fill="#1a1a1a">Input layer</text>'
    )
    parts.append(
        f'<text x="{in_x}" y="65" text-anchor="middle" '
        f'font-size="13" fill="#5F5E5A">{len(inputs)} nodes</text>'
    )
    parts.append(
        f'<text x="{hid_x}" y="45" text-anchor="middle" '
        f'font-size="17" font-weight="bold" fill="#1a1a1a">Hidden layer</text>'
    )
    parts.append(
        f'<text x="{hid_x}" y="65" text-anchor="middle" '
        f'font-size="13" fill="#5F5E5A">128 neurons (ReLU), fully connected</text>'
    )
    parts.append(
        f'<text x="{out_x}" y="45" text-anchor="middle" '
        f'font-size="17" font-weight="bold" fill="#1a1a1a">Output layer</text>'
    )
    parts.append(
        f'<text x="{out_x}" y="65" text-anchor="middle" '
        f'font-size="13" fill="#5F5E5A">{len(outputs)} nodes</text>'
    )

    # ---- bottom caption ----
    parts.append(
        f'<text x="{W/2}" y="{H-25}" text-anchor="middle" '
        f'font-size="13" fill="#5F5E5A">{title}</text>'
    )

    parts.append('</svg>')
    svg = "\n".join(parts)
    with open(fname, "w", encoding="utf-8") as f:
        f.write(svg)
    return fname, len(parts)


f1 = build_svg(
    inputs=["e(t)", "\u0394e(t)"],
    outputs=["kp", "ki", "kd"],
    fname="nn_2input_128_3output.svg",
    title="2 inputs \u2192 128 hidden \u2192 3 outputs   (256 + 384 = 640 weighted connections)",
)

f2 = build_svg(
    inputs=["Power level", "e(t)", "\u0394e(t)"],
    outputs=["kp", "ki", "kd"],
    fname="nn_3input_128_3output.svg",
    title="3 inputs \u2192 128 hidden \u2192 3 outputs   (384 + 384 = 768 weighted connections)",
)

print(f1)
print(f2)
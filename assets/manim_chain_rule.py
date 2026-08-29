"""The chain rule, on tape: how ember's backward() actually works.

Animates the exact computation from the README's hero graph: a two-layer
network's forward pass flowing left to right, then the gradient igniting at
the loss and burning backward through the tape, composing local derivatives
as it goes. Full LaTeX typography via MathTex. Ember palette throughout.

    manim -qh --format=mp4 assets/manim_chain_rule.py ChainRuleOnTape
"""

from manim import (
    config, Scene, VGroup, Circle, Text, MathTex, Arrow, Dot, FadeIn, FadeOut,
    Create, TransformMatchingTex, LaggedStart, Flash, RIGHT, LEFT, UP, DOWN,
    rate_functions, interpolate_color, ManimColor,
)

BG = "#191512"
INK = "#e8ded2"
FAINT = "#8a7d6d"
GRID = "#4a4038"
ASH = "#6b7f8c"
HEAT = ["#7f1d0e", "#c2410c", "#f59e0b", "#fde68a", "#fffbeb"]

config.background_color = BG

MONO = "Menlo"


def heat_color(t):
    """t in [0,1] -> ember heat, interpolated across the palette stops."""
    t = max(0.0, min(1.0, t)) * (len(HEAT) - 1)
    i = min(int(t), len(HEAT) - 2)
    return interpolate_color(ManimColor(HEAT[i]), ManimColor(HEAT[i + 1]), t - i)


class ChainRuleOnTape(Scene):
    def construct(self):
        # ---- the tape: x -> matmul W1 -> +b1 -> relu -> matmul W2 -> +b2 -> L
        op_tex = [r"x", r"W_1", r"b_1", r"\mathrm{relu}", r"W_2", r"b_2", r"L"]
        n = len(op_tex)
        xs = [-5.4 + i * 1.8 for i in range(n)]
        nodes, labels = VGroup(), VGroup()
        for i, (op, x) in enumerate(zip(op_tex, xs)):
            c = Circle(radius=0.34, color=GRID, fill_color=BG,
                       fill_opacity=1.0, stroke_width=2.5).move_to([x, 0.9, 0])
            t = MathTex(op, font_size=30, color=INK)
            t.next_to(c, DOWN, buff=0.22)
            nodes.add(c)
            labels.add(t)
        edges = VGroup(*[
            Arrow(nodes[i].get_right(), nodes[i + 1].get_left(),
                  buff=0.06, color=GRID, stroke_width=2.5,
                  max_tip_length_to_length_ratio=0.12)
            for i in range(n - 1)
        ])

        title = Text("backward() is just the chain rule, run in reverse",
                     font=MONO, font_size=26, color=INK).to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(VGroup(c, t), shift=RIGHT * 0.3)
                                for c, t in zip(nodes, labels)],
                              lag_ratio=0.12),
                  LaggedStart(*[Create(e) for e in edges], lag_ratio=0.12),
                  run_time=2.2)

        # ---- forward: a value pulse flows left to right
        fwd_note = Text("forward: values flow, every op lands on the tape",
                        font=MONO, font_size=20, color=FAINT).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(fwd_note), run_time=0.6)
        pulse = Dot(color=ASH, radius=0.11).move_to(nodes[0].get_center())
        self.add(pulse)
        for i in range(1, n):
            self.play(pulse.animate.move_to(nodes[i].get_center()),
                      run_time=0.34, rate_func=rate_functions.ease_in_out_sine)
            nodes[i].set_stroke(ASH)
        self.play(FadeOut(pulse), run_time=0.3)

        # ---- ignition at the loss
        self.play(FadeOut(fwd_note), run_time=0.4)
        seed = MathTex(r"\frac{\partial L}{\partial L} = 1",
                       font_size=34, color=heat_color(0.95)).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(seed), run_time=0.5)
        self.play(nodes[-1].animate.set_fill(heat_color(1.0)).set_stroke(heat_color(1.0)),
                  Flash(nodes[-1], color=heat_color(0.9), line_length=0.25),
                  run_time=0.8)

        # ---- gradient burns backward, composing local derivatives
        chain_tex = [
            r"\frac{\partial L}{\partial z_2}",
            r"\frac{\partial L}{\partial z_2}\cdot\frac{\partial z_2}{\partial h}",
            r"\frac{\partial L}{\partial z_2}\cdot\frac{\partial z_2}{\partial h}"
            r"\cdot \mathrm{relu}'(z_1)",
            r"\frac{\partial L}{\partial z_2}\cdot\frac{\partial z_2}{\partial h}"
            r"\cdot \mathrm{relu}'(z_1)\cdot\frac{\partial z_1}{\partial x}",
        ]
        self.play(FadeOut(seed), run_time=0.3)
        formula = MathTex(chain_tex[0], font_size=34,
                          color=heat_color(0.9)).move_to([0, -1.7, 0])
        step_targets = [5, 4, 3, 1]  # node indices the gradient reaches
        for k, upto in enumerate(step_targets):
            burn = []
            lo = step_targets[k]
            hi = step_targets[k - 1] if k > 0 else n - 1
            for j in range(hi - 1, lo - 1, -1):
                burn.append(edges[j].animate.set_color(heat_color(0.9 - 0.12 * k)))
                burn.append(nodes[j].animate.set_fill(
                    heat_color(0.85 - 0.12 * k)).set_stroke(heat_color(0.9 - 0.12 * k)))
            if k == 0:
                self.play(*burn, FadeIn(formula), run_time=1.1)
            else:
                new_formula = MathTex(chain_tex[k], font_size=34,
                                      color=heat_color(0.9 - 0.12 * k)
                                      ).move_to([0, -1.7, 0])
                self.play(*burn, TransformMatchingTex(formula, new_formula),
                          run_time=1.2)
                formula = new_formula
            self.wait(0.25)

        # ---- the accumulation point
        acc = MathTex(
            r"\text{paths that meet, sum: }\quad"
            r"\frac{\partial L}{\partial \theta} \mathrel{+}= g",
            font_size=32, color=FAINT,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(acc), run_time=0.7)
        self.play(Flash(nodes[1], color=heat_color(0.7), line_length=0.3),
                  Flash(nodes[4], color=heat_color(0.7), line_length=0.3),
                  run_time=0.9)
        self.wait(0.6)

        # ---- end card
        self.play(FadeOut(formula), FadeOut(acc), run_time=0.6)
        end = VGroup(
            Text("that is all of deep learning's training signal:",
                 font=MONO, font_size=22, color=INK),
            MathTex(r"\text{local derivatives, multiplied along the tape,}",
                    font_size=30, color=heat_color(0.75)),
            MathTex(r"\text{summed where paths meet}",
                    font_size=30, color=heat_color(0.75)),
            Text("ember does it in 700 lines of NumPy", font=MONO,
                 font_size=18, color=FAINT),
        ).arrange(DOWN, buff=0.28).move_to([0, -1.6, 0])
        self.play(FadeIn(end, shift=UP * 0.2), run_time=1.0)
        self.wait(1.6)

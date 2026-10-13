你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="332f06d7" lines="9-23">
<description>
Slide a solid square through a winding channel to reach a same-sized target square.

Input structure:
- bg (most common color). A river color forms a winding corridor.
- Two same-sized NxN solid squares on the river: mover and destination, each a distinct color.

Rule:
- The mover square travels along the river toward the destination.
- It can only pass through positions where the full NxN area is river.
- If reachable, the mover replaces the destination; otherwise it stops at the farthest reachable point.
- The mover's original position is filled with the river color.

Key insight: Chokepoints narrower than N pixels block the square, forcing it to stop short.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="a6f40cea" lines="74-88">
<description>
Complete partial paths inside a framed rectangle by projecting external color traces inward.

Input structure:
- bg, a 1px rectangular frame (border color) with uniform interior fill.
- Colored components outside the frame touching its edges.

Rule:
- Output = the frame's interior, filled with the fill color.
- Map each outside component's frame-touch points into interior coordinates.
- Monochrome components become solid Manhattan stripes between mapped endpoints.
- Two-color components become alternating-color Manhattan paths.

Key insight: Endpoint colors of two-color traces anchor the alternating phase deterministically.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="bf45cf4b" lines="7-19">
<description>
Scale up a monochrome control sprite by stamping a polychrome tile at each filled pixel.

Input structure:
- bg = background color.
- Two foreground objects (8-connected): a monochrome rectangular sprite with bg holes (control), and a polychrome square tile (block).

Rule:
- For each cell in the control sprite's bbox: if filled, stamp a copy of the block tile; if bg, leave a tile-sized bg region.
- Output grid dimensions are (control_h × tile_h) by (control_w × tile_w).

Key insight: The control sprite acts as a binary mask that gets "pixel-scaled" by the tile — each pixel expands to a full tile copy or empty space.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="981571dc" lines="6-19">
<description>
Reconstruct a 4-way symmetric mosaic by filling missing (black) pixels.

Input structure:
- A colored grid exhibiting bilateral symmetry along both a vertical and a horizontal axis.
- Some pixels are erased to color 0 (black), creating gaps in the pattern.

Rule:
- Infer the vertical and horizontal reflection axes from the visible (non-zero) pixels.
- For each black pixel, recover its color from the mirrored position across either axis or both.
- When multiple mirrors yield candidates, use the majority color.

Key insight: The reflection axes are not given — they must be deduced from the surviving pixels by finding the axes that best preserve the existing symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="320afe60" lines="41-56">
<description>
Slide each foreground object horizontally to the edge indicated by whether its shape has an open notch.

Input structure:
- Yellow is the background and there is one blue foreground color.
- The foreground is split into separated 4-connected objects.
- Each object keeps its row span and shape in the output.

Rule:
- If an object's bbox border contains a background gap flanked by object pixels along that same border, slide it to the right edge and recolor it green.
- Otherwise slide the object to the left edge and recolor it red.
- Start from a blank background canvas; no moved objects overlap.

Key insight:
- The target side is encoded by the object's shape, not by its original position: open notches go right, while closed or solid shapes go left.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="9aaea919" lines="38-51">
<description>
Transfer sprites from red-marked lanes to green-marked lanes, recoloring and stacking.

Input structure:
- bg = most common color. Red (2) and green (3) thin bars define vertical lanes.
- Each lane contains repeated identical monochrome sprites in an evenly spaced vertical grid.

Rule:
- Each red lane's sprites turn grey (5); their count transfers to the nearest green lane.
- New sprites fill empty lattice slots bottom-up, using the source shape recolored to the destination lane's sprite color.
- All marker bars are removed.

Key insight: Sprite count is conserved across lanes — red sprites become grey, and equally many new sprites stack into the green lane's regular grid.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="465b7d93" lines="97-117">
<description>
Move the small colored sprite into the hollow frame by stretching its shape to the frame interior.

Input structure:
- The background is the dominant color.
- There are exactly two foreground colors: one forms a complete rectangular
  one-pixel-thick frame, and the other forms one 4-connected sprite outside it.
- The frame interior is initially background.

Rule:
- Clear the original sprite from its outside location.
- Map the sprite bbox to the frame interior bbox, sending bbox extremes to
  interior extremes.
- Draw straight horizontal or vertical segments between mapped positions of
  adjacent sprite pixels, then fill any enclosed background region inside
  those drawn segments.

Key insight:
The outside sprite is a miniature line/fill diagram for the frame interior,
not a bitmap to scale by block replication.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="3dc255db" lines="37-53">
<description>
Eject content pixels from cup-shaped frames outward along each cup's tail direction.

Input structure:
- bg = background. Hollow cup/Y/funnel frames, each a unique border color.
- Each cup has a convergent tail spike on one side and an open mouth on the opposite side.
- Content pixels (a single non-border color per cup) sit inside the cup's mouth interior.

Rule:
- For each cup, identify its tail spike and tail direction.
- Erase all content pixels from the mouth interior.
- Draw a straight line of content-colored pixels from the spike outward in the tail direction,
  one pixel per erased content pixel; cells beyond the grid edge are dropped.

Key insight: The tail spike is the unique convergent point whose outward direction is opposite
the mouth opening where the content lives.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="18286ef8" lines="17-31">
<description>
Use the pink control pixel to choose where the maroon pixel belongs inside the grey 3x3 region, and turn the control pixel maroon.

Input structure:
- The grid contains one 3x3 region made from eight grey pixels and one maroon pixel.
- There is exactly one pink control pixel outside that region.

Rule:
- Restore the old maroon cell in the 3x3 region to grey.
- Compare the pink pixel to the 3x3 region: above/same band/below selects top/middle/bottom, and left/same band/right selects left/middle/right.
- Paint the selected cell maroon and recolor the pink control pixel maroon.

Key insight:
- The pink pixel acts as a relative-position marker for a 3x3 target inside the grey region.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="db118e2a" lines="7-23">
<description>
Normalize the framed input motif into a 5x5 icon and stamp two copies on a 15x15 canvas.

Input structure
The input is a rounded rectangular frame: the four corners are the background
color, and every other outer-border pixel has one frame color. Non-background
pixels inside that frame are the motif to preserve.

Rule
Make a 5x5 version of the rounded frame with a 3x3 interior. Crop the input
interior motif to its bounding box, center that crop in the 3x3 interior, and
paste the resulting 5x5 icon at upper-left positions (2, 2) and (8, 8) in a
15x15 background canvas.

Key insight
The variable-size frame is only a carrier; the meaningful object is the
compact interior motif, which is normalized to the fixed 3x3 icon interior.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="8b9c3697" lines="9-23">
<description>
Insert red components into interior sockets of large structures, leaving a black trail.

Input structure:
- bg = most common. Large non-bg, non-red structures with interior cavities.
- Red components outside structures.

Rule:
- Interior sockets: bg cells inside a structure's bbox, adjacent to structure pixels.
- For each structure, find the direction where rays from sockets hit a red component.
- Slide that component into the socket; paint traversed cells black.
- Unmatched reds vanish.

Key insight: Docking direction is inferred by ray-casting from interior sockets to the nearest red component.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="37ce87bb" lines="33-47">
<description>
Compute a grey result bar by adding teal bar heights and subtracting red bar heights.

Input structure
- The grid has a dominant background and several bottom-aligned one-cell-wide vertical bars.
- Teal bars are positive operands; red bars are negative operands.
- The penultimate column is an empty result slot.

Rule
- Keep every input bar unchanged.
- Compute result height = total teal bar height - total red bar height.
- Paint that many bottom cells of the penultimate column grey.

Key insight
- The bars encode arithmetic by height, with color providing the sign.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="a1aa0c1e" lines="74-97">
<description>
Compress the stacked colored bars into a compact ordered table with the control pixel on the selected row.

Input structure
- A background grid contains several full-width colored bars from top to
  bottom; the last bar is the base/separator color.
- Each non-base bar may have an aligned three-column comb motif below it,
  alternating side-pair rows with solid rows.
- A singleton control pixel sits away from the colored motifs.

Rule
- Make one output row per non-base bar, preserving top-to-bottom order.
- In that row, paint the bar color once for each solid three-pixel row in
  its motif, then paint the base-bar color in the separator column.
- The singleton control pixel sits immediately above the base bar. Among
  motifs whose open side-pair parity is compatible with that bottom row,
  put the control pixel in the output row with the fewest solid rows; use
  horizontal distance to the motif center only to break count ties.

Key insight
- The vertical comb height is a count, while the control pixel is read by
  projecting it onto the center gap of compatible comb rows rather than by
  its vertical band alone.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="1a244afd" lines="49-68">
<description>
Rotate each magenta source pixel 90 degrees counterclockwise around its matched blue pivot, recoloring the new pixel orange.

Input structure:
- The background is the dominant color.
- The only foreground colors are blue pivot pixels and magenta source pixels.
- Each magenta source can be paired one-to-one with a blue pivot in the same
  row or column.

Rule:
- Remove all magenta source pixels.
- For each matched source/pivot pair, preserve the source distance from the
  pivot but rotate its offset 90 degrees counterclockwise.
- Paint the rotated in-bounds target orange; if the target is outside the
  grid, the source simply disappears.

Key insight:
The blue pixels are rotation centers, and row/column alignment gives a
unique global matching from each magenta source to its pivot.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="e12f9a14" lines="138-152">
<description>
Emit diagonal beams from framed 2x2 blocks through frame gaps; beams interact on collision.

Input structure:
- bg = most common color. Wall color (e.g. green) forms 4x4 frames around 2x2 monochrome blocks.
- Each frame has one or more gap pixels where the wall is missing.

Rule:
- Each gap emits a beam of the inner block's color, traveling diagonally outward (direction set by gap position relative to the block).
- Beams paint every cell they cross until they hit the grid edge or collide.
- 90° collision: both beams turn 45° toward their bisector direction; the intersection cell is left blank (1-pixel gap).
- 45° collision: beams merge into a zigzag (alternating cardinal + diagonal steps) with no gap at the meeting point.

Key insight: Two distinct collision rules — 90° produces a gap and single-pixel diagonal lines; 45° produces no gap and thicker zigzag bands.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="6ffbe589" lines="83-98">
<description>
Crop a board cluster and rotate its color layers based on outside cue sprites.

Input structure:
- bg = most common color.
- A main multi-color board cluster (largest foreground region).
- Small monochrome cue sprites outside the board for the colors that rotate.

Rule:
- Tight-crop around the board cluster.
- For each color layer inside the crop, find its same-color cue outside.
- Rotate that layer clockwise by (cue pixel count mod 4) × 90°.
- No outside cue → no rotation.

Key insight: Cue sprite size mod 4 encodes rotation amount for its matching color layer.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="e729b7be" lines="8-20">
<description>
Complete the scene by adding the point-reflected copy of the colored object across the central marker.

Input structure
- The grid has odd dimensions, a dominant background, and a center-row marker color appearing only at the left edge, center, and right edge.
- A single divider color fills the center column except for the center marker.
- All remaining foreground pixels form the object to copy, entirely on the left side of the divider.
Rule
- Keep the input unchanged.
- For every object pixel, paint the pixel rotated 180 degrees around the center marker with the same color.
Key insight
- The center marker is the rotation center; the yellow-looking divider is not mirrored, it only separates the source from the empty target side.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="f8cc533f" lines="10-25">
<description>
Complete every colored sprite by applying the union of all aligned partial sprite masks.

Input structure:
- The dominant color is the background.
- Each non-background color appears as a separated partial copy of one common sprite mask.
- The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

Rule:
- Crop each same-color mask to its bounding box.
- Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
- Fill that template at every sprite location using the sprite's own color.

Key insight:
- No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="6e453dd6" lines="39-52">
<description>
Slide black shapes rightward to a grey separator, then shoot red rays from enclosed holes.

Input structure:
- A vertical grey (5) separator column divides the grid into left and right halves.
- Black shapes sit on the left side against a background color (bg).

Rule:
- Translate each black connected component rightward until its right edge is flush with the separator.
- After sliding, find enclosed bg regions (holes) within the black shapes that touch the column just left of the separator.
- For each row spanned by such a hole, emit a red ray across the entire right half of the grid.

Key insight: Only holes adjacent to the separator trigger rays — the sliding creates enclosures that act as "nozzles" projecting rightward.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="758abdf0" lines="54-69">
<description>
Advance short marker bars away from the full edge bar, turning completed two-pixel markers into edge-color marks at the opposite side.

Input structure
- The background is the dominant color.
- One non-background color forms a complete one-cell-thick bar on exactly one outer edge.
- The other foreground color appears only in the first two cells inward from that bar, lane by lane.

Rule
- In each lane perpendicular to the edge bar, a one-pixel marker adjacent to the bar grows one more cell inward.
- A two-pixel marker in those first two inward cells is cleared back to background.
- The cleared two-pixel marker is replaced by a two-pixel mark of the edge-bar color at the opposite side of the same lane.

Key insight
- The full edge bar defines both the orientation and the direction of the local two-step marker cycle.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

请一次性生成 5 个 `<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 在符合ARC题目和提示词要求的基础上，鼓励尝试创新规则为多层结构的题目，不局限于两层，可以创新三层或四层结构的题目，但不可为了多结构层数违背要求。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。


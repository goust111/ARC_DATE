你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="8dab14c2" lines="120-133">
<description>
Complete each boundary tab or notch by adding the complementary feature on the opposite side of the foreground object.

Input structure
- The grid has exactly two colors: a dominant background and one foreground object.
- The foreground object is a thick rectilinear shape with one-pixel tabs and notches on its boundary.

Rule
- A foreground tab is a one-neighbor boundary pixel; project through its full row or column span and remove the opposite edge pixel to make a matching notch.
- A background notch is a one-pixel cut with three foreground 4-neighbors; project across the full row or column span and add a tab just outside the opposite edge.
- Notches immediately beside a tab source are not independent sources, because that gap belongs to the tab's local boundary shape.

Key insight
- The changed pixels are not color-specific noise; they are paired jigsaw-like boundary features on opposite sides of the same thick row or column cross-section.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="88e364bc" lines="102-115">
<description>
Slide yellow pixels inside walled regions in the direction dictated by border-legend arrows.

Input structure:
- bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
- Border legends: small rectangles on the grid edge whose fill color C matches a region's border color. Each legend contains a center dot and an indicator dot encoding a direction.

Rule:
- For each enclosed region, identify its border color C and look up the legend with fill color C.
- The direction from the legend's center dot to its indicator dot gives the slide direction.
- All yellow pixels in that region slide in that direction, stacking against the far wall.

Key insight: Legend fill color links to region border color; the center→indicator offset encodes the movement vector.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="3d588dc9" lines="95-109">
<description>
Replace the active black cutout's tail with background and mark the cutout side facing a colored source region.

Input structure:
- The dominant color is background, with black rectangle-plus-tail cutouts, one gray source region, and colored distractors.
- Non-gray colored regions and single pixels are distractors/blockers, even when they are multi-pixel staircase shapes.

Rule:
- Find each black cutout whose largest solid rectangle leaves exactly a two-pixel tail.
- The active gray source region is visible only by horizontal rays through background from a cutout's left or right side.
- Select the unique cutout side with the most same-source visible rows, erase that cutout's tail to background, and recolor the full selected vertical side pink (6).

Key insight:
- The colored regions do not move; their horizontal line-of-sight identifies which rectangular cutout is being opened and which side receives the pink mark.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="db7260a4" lines="37-56">
<description>
Drop the singleton marker into the matching red U-shaped container, or make a bottom-row puddle if no closed container is hit.

Input structure
- The grid has a dominant background, one singleton marker pixel on the top row,
  and one other foreground color forming vertical bars and bottom segments.
- A closed container is a U shape: two vertical side bars with a continuous
  bottom segment between them and an empty rectangular interior above the bottom.

Rule
- Remove the top marker from its original cell.
- If exactly one closed U-shaped container has the marker column within its
  horizontal footprint, fill that container's empty interior with the marker color.
- If no closed container is under the marker column, fill the whole bottom row
  with the marker color.

Key insight
- The marker does not identify a color by value; it identifies a vertical drop
  column, and only a structurally closed U shape can catch and contain the fill.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="e9fc42f2" lines="26-43">
<description>
Assemble the foreground objects by docking equal-colored boundary markers edge-to-edge.

Input structure
The grid has a dominant background color and several 4-connected multicolor foreground
objects.  One foreground color is the body color; the less frequent foreground colors are
boundary connector markers, and each marker color appears once as a free end or twice as a
matched docking pair.

Rule
Crop each foreground object to its bounding box and translate the objects so that the two
occurrences of every paired connector color touch on their exposed boundary sides.  Preserve
each object's orientation and colors, then return the tight crop around the assembled
foreground on the original background color.

Key insight
The colored marker pixels are ports: matching colors specify which objects must touch, and
the marker's side of its object determines the direction of the translation.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="1ae2feb7" lines="30-45">
<description>
Extend row bars periodically across the opposite side of a separator column.

Input structure:
- bg = background. A near-solid vertical separator divides the grid into two sides.
- One side contains horizontal colored bars; the other side is blank target space.
- Bars may be stacked by row and may sit at different distances from the separator.

Rule:
- Identify the separator as the column whose non-bg cells are one color.
- Treat the denser side as the bar side and collect each contiguous row run.
- Process bars from nearest to farthest from the separator; for each bar, paint its
  color every bar-length cells into the blank side without overwriting earlier paint.

Key insight: Each bar's length is the period for its extension, and nearer bars claim target cells first.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="2d0172a1" lines="209-224">
<description>
Convert a rough nested line drawing into a clean canonical frame-and-dot diagram.

Input structure:
- The grid has one background color and one foreground color.
- Foreground 4-connected objects are rough closed frames or small filled blobs.
- Frames may contain other frames/blobs; blobs outside a frame remain outside it.

Rule:
- Detect the containment tree of foreground objects using holes of each frame object.
- Replace every blob with one foreground pixel.
- Replace every frame with a rectangular border padded two cells around its direct children.
- Place siblings by their original row/column overlap and order, with one empty cell between canonical objects.

Key insight: The exact wobbly geometry is irrelevant; only enclosure plus relative sibling positions define the output.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="11dc524f" lines="25-40">
<description>
Slide the non-square sprite to the square anchor and replace the anchor with the sprite's mirror image.

Input structure:
- The background is the most common color.
- There is one 8-connected non-background sprite that is not a solid 2x2 square.
- There is one solid 2x2 square anchor in another non-background color.

Rule:
- Move the sprite straight toward the anchor along their unique cardinal separation until it touches the anchor's near edge.
- Reflect the moved sprite across that near edge and draw the reflected copy in the anchor color.
- Clear the original locations; the output keeps only the moved sprite and reflected anchor-color copy.

Key insight:
- The 2x2 square marks the mirror line: its side facing the sprite becomes the contact edge for a two-color reflected pair.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="52364a65" lines="10-24">
<description>
Erase the two leftmost columns of every foreground object, leaving the rest of each object in place.

Input structure:
- A grid has one dominant background color and several monochromatic 4-connected foreground objects.
- Each foreground object spans at least two columns in its bounding box.

Rule:
- Find each same-color 4-connected foreground object.
- Within each object's own bounding box, set foreground pixels in the two leftmost columns to the background color.
- Leave all other pixels unchanged.

Key insight:
- The deletion is measured from each object's local left edge, not from the grid edge or from a shared global column.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="470c91de" lines="48-65">
<description>
Move each teal-cornered filled rectangle one cell diagonally toward its teal corner.

Input structure:
- Dominant background surrounds several filled rectangular colored objects.
- Each object has exactly one teal control pixel replacing one corner cell of
  its rectangle.

Rule:
- Treat the teal pixel as the missing corner of its colored rectangle.
- Erase each original rectangle footprint, including the teal pixel.
- Repaint the completed rectangle shifted one cell diagonally in the direction
  of that teal corner.

Key insight:
- The control pixel's corner position gives both the missing cell to restore
  and the diagonal motion vector.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="1a244afd" lines="49-68">
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

<reference_puzzle index="13" source="eval" puzzle_id="3dc255db" lines="37-53">
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

<reference_puzzle index="14" source="train" puzzle_id="46c35fc7" lines="21-34">
<description>
Rotate the colors around every 3x3 foreground ring, with corners and side-centers cycling in opposite directions.

Input structure
- The grid has a dominant background color.
- Each 4-connected foreground object is a complete 3x3 ring: eight foreground pixels around a background center.

Rule
- For each ring, move corner colors one step around the four corners.
- Move side-center colors one step around the four side centers in the opposite direction.
- Leave the background center and all other background pixels unchanged.

Key insight
- Corners and side centers form two separate four-cycles on the same 3x3 ring.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="332202d5" lines="49-64">
<description>
Fill every row from the nearest marked horizontal bar, turning bar rows and color-boundary ties into separator rows.

Input structure
- A dominant background fills the grid except for full-width horizontal bars and one vertical marker column.
- Outside the marker column, every row is monochrome: either background or a source bar color.
- Background rows share one marker color in the marker column; source bar rows share the other.

Rule
- Each non-source row takes the color of the nearest source bar, with that row's marker-column pixel set to the source-marker color.
- If a row is exactly midway between nearest bars of different colors, make the whole row the source-marker color.
- Each original source bar row becomes a source-marker row with the marker-column pixel changed to the background-row marker color.

Key insight
- The marked bars are horizontal seeds for a one-dimensional nearest-seed partition over rows; equal-distance rows are separators only when the competing seed colors differ.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="78332cb0" lines="10-23">
<description>
Reorder 5×5 blocks into a connected strip by matching edge foreground pixels.

Input structure:
- bg = background; sep = divider color (full row/column lines).
- Several 5×5 sprite blocks in a grid, separated by sep lines.

Rule:
- Two blocks "connect" when their facing edges share foreground pixels at the same offset (e.g., A's right edge ∩ B's left edge).
- Chain all blocks in the orientation (horizontal or vertical) that admits a complete sequence.
- Output is a 1×N strip of blocks in chain order, separated by sep lines.

Key insight: Edge-contact overlap determines both block ordering and output orientation.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="269e22fb" lines="46-60">
<description>
Complete a visible crop into the shared 20x20 binary motif in its matching orientation.

Input structure:
- The grid is a two-color rectangular crop of one rotation/reflection of the same 20x20 motif.
- One color is the motif foreground and the other color is the motif background.

Rule:
- Try each input color as the foreground mask.
- Match that mask against every possible window in every dihedral orientation of the motif.
- Repaint the uniquely matched full 20x20 motif with the matched foreground/background colors.

Key insight:
- The examples are partial windows of one canonical binary pattern; the local crop is distinctive enough to identify the full orientation and offset.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="902510d5" lines="58-80">
<description>
Keep the largest 8-connected motif and replace the corner/control pixels with a solid corner triangle.

Input structure
- The background is the dominant color.
- Exactly one grid corner contains a foreground control pixel; it gives the
  corner from which the triangle is painted.
- There is one unique largest 8-connected monochrome foreground object, the
  motif that remains in the output; all other foreground pixels are controls.
- Every foreground component outside that motif is a singleton control pixel.
- The fill color is the unique control color that appears in at least two
  non-corner control pixels.

Rule
- Start from a blank background canvas and redraw only the largest motif.
- Count the non-corner control pixels; that count is the triangle side.
- Paint the fill color on the right-triangle region where those two distances
  from the marked corner's edges sum to less than the side length.

Key insight
- The scattered singleton controls are counted into a compact triangular
  stack at the marked corner; only the largest motif is not part of the count.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="50c07299" lines="8-21">
<description>
Move the diagonal foreground segment to the next longer run along the same up-right diagonal.

Input structure
The grid has one background color and one foreground color. The foreground pixels form a
contiguous one-pixel-wide segment on a single anti-diagonal.

Rule
Erase the current segment. If its columns are c0..c1 and its length is L, draw the same
color on columns c1+1..c1+L+1 along the same anti-diagonal.

Key insight
The segment advances past its current upper-right end by one whole segment position, and
the next segment is one pixel longer than the current one.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="e3fe1151" lines="36-55">
<description>
Fill the one blank in each corner region so all four regions have the same color multiset.

Input structure:
- The dominant background color forms one solid separator row and one solid
  separator column.
- The separator cross splits the grid into four equal rectangular corner regions.
- Each corner region contains exactly one background-colored blank pixel.

Rule:
- Count the non-background colors in each corner region.
- The completed region multiset is defined by the maximum observed count of
  each color across the four regions.
- Replace each region's blank pixel with the single color needed to match that
  completed multiset.

Key insight:
- The separator color inside a corner region is a missing value marker; comparing
  the four regions reveals the shared multiset without using numeric color order.
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
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id是不可用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
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
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。

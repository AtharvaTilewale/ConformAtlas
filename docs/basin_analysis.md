# Basin Analysis & Representative Structure Extraction

ConformAtlas automatically detects energy basins and extracts verified full-atom structures corresponding to each conformational state.

---

## 1. Topological Watershed Segmentation

Rather than simply finding local minimum pixels, ConformAtlas uses **topological watershed segmentation**:
1. Candidate minimum energy seeds are identified as local peaks in probability density.
2. The watershed algorithm segments the continuous energy landscape into catchment basins bounded by energy ridges.
3. Every occupied grid cell is assigned to its catchment basin.
4. Frames are assigned to the basin corresponding to their $(PC_1, PC_2)$ coordinates.

---

## 2. Conformational States

* **State Labeling**: Basins are sorted by population descending and labeled `State A` (dominant state), `State B` (second dominant), etc.
* **Population Filtering**: Basins containing $\ge 5\%$ of frames (`--min-state-population 0.05`) are classified as major states. Low-occupancy basins are collected into the unassigned/minor fraction.
* **Population Sum**: $\sum \text{Major States} + \text{Unassigned} = 100\%$.

---

## 3. Representative Structure Extraction

For each state, ConformAtlas extracts a verified real frame from the original full trajectory:
1. Identifies the minimum free energy coordinates $(PC_{1, \min}, PC_{2, \min})$ of the basin.
2. Identifies all frames assigned to this state.
3. Computes the Euclidean distance $d = \sqrt{(PC_1 - PC_{1, \min})^2 + (PC_2 - PC_{2, \min})^2}$.
4. Selects the frame with minimal distance and writes `states/State_X/representative.pdb`.
5. Writes `states/State_X/metadata.json` recording the exact replicate, frame index, time (ps/ns), and coordinates.

# Feature dictionary

340 candidate features per patient (287 tabular + 53 image); 203 retained after gain-based
selection. Definitions of raw FreeSurfer regions and NACC variables defer to FreeSurfer and
the NACC Researcher's Data Dictionary.

## FreeSurfer cortical parcellation (186)

`{LH|RH}_{REGION}_{SUFFIX}`, where SUFFIX = **GVOL** (gray-matter volume, mm³),
**SAREA** (surface area, mm²), or **AVGTH** (mean thickness, mm); 31 regions × 2 hemispheres
× 3 measures.

Regions: caudalanteriorcingulate, caudalmiddlefrontal, cuneus, entorhinal, fusiform,
inferiorparietal, inferiortemporal, insula, isthmuscingulate, lateraloccipital,
lateralorbitofrontal, lingual, medialorbitofrontal, middletemporal, paracentral,
parahippocampal, parsopercularis, parsorbitalis, parstriangularis, pericalcarine, postcentral,
posteriorcingulate, precentral, precuneus, rostralanteriorcingulate, rostralmiddlefrontal,
superiorfrontal, superiorparietal, superiortemporal, supramarginal, transversetemporal.
(The Desikan–Killiany bankssts, frontalpole, and temporalpole labels are excluded.)

## FreeSurfer subcortical & global volumes (47)

Segmented volume (mm³) per structure, in four groups:

- **Global (17):** BRAINSEG, CORTEX, TOTALGRAY, SUBCORTGRAY, LHCORTEX, RHCORTEX,
  CEREBRALWHITEMATTER, LHCEREBRALWHITEMATTER, RHCEREBRALWHITEMATTER, SUPRATENTORIAL,
  SUPRATENTORIALNOTVENT, ESTIMATEDTOTALINTRACRANIALVOL, CEREBRUMTCV, GM, CSF, WMH,
  WM_HYPOINTENSITIES
- **Subcortical (15):** LEFT/RIGHT × {thalamus proper, caudate, putamen, pallidum, amygdala,
  accumbens area, ventral DC}, plus BRAIN_STEM
- **Ventricular (6):** LEFT/RIGHT_LATERAL_VENTRICLE, LEFT/RIGHT_INF_LAT_VENT, 3RD_VENTRICLE,
  4TH_VENTRICLE
- **Hippocampal (9):** HIPPOCAMPUS, LEFT/RIGHT_HIPPOCAMPUS, LEFT/RIGHT_HIPPO,
  LH/RH_PARAHIPPOCAMPAL_GVOL, LH/RH_ENTORHINAL_GVOL

Counts reflect model-input columns. `LH/RH_ENTORHINAL_GVOL` and `LH/RH_PARAHIPPOCAMPAL_GVOL`
also appear under cortical parcellation, and `LEFT/RIGHT_HIPPO` duplicate the hippocampus
volumes (6 redundant columns).

## Hand-crafted image features (53)

`hc_s{S}_{METRIC}` (S = axial slice 1–5): brain_frac, mean_int, std_int, median_int, csf_frac,
gray_frac, white_frac, ventricle_ratio, asymmetry, edge_density (50 per-slice). Tissue
fractions use fixed normalized-intensity thresholds (CSF 0.05–0.20, gray 0.20–0.35,
white ≥ 0.35). Plus 3 cross-slice features: hc_slice_int_var, hc_slice_brain_var,
hc_slice_brain_range.

## Clinical (11)

Demographics: NACCAGE, SEX, EDUC, NACCNE4S (APOE ε4 allele count, 0–2). Comorbidities: HYPERT,
HYPERCHO, DIABETES, CVHATT, CBSTROKE, DEPD. Syndrome flag: AMNDEM.

## Rate-of-change (22)

`{STRUCTURE}_ROC` = (latest visit − earliest visit) / years elapsed. Structures: HIPPOCAMPUS,
LEFT/RIGHT_HIPPOCAMPUS, LEFT/RIGHT_LATERAL_VENTRICLE, TOTALGRAY, CEREBRALWHITEMATTER, BRAINSEG,
LEFT/RIGHT_AMYGDALA, LH/RH_ENTORHINAL_GVOL, LH/RH_ENTORHINAL_AVGTH, LH/RH_FUSIFORM_GVOL,
LH/RH_MIDDLETEMPORAL_GVOL, LH/RH_INFERIORTEMPORAL_GVOL, WMH, CSF.

## Interaction (21)

- **Ratios (a / b, 8):** HIPPO_BRAIN, HIPPO_ICV, VENT_BRAIN, VENT_ICV, CSF_BRAIN, WMH_WM,
  GRAY_WHITE, CORTEX_BRAIN
- **Asymmetries (|L − R| / (L + R), 11):** HIPPO, AMYGDALA, VENT, THALAMUS, CAUDATE, PUTAMEN,
  ENTORHINAL_VOL, ENTORHINAL_TH, FUSIFORM, INFTEMP, MIDTEMP
- **Age ratios (structure / NACCAGE, 2):** HIPPO_AGE, BRAIN_AGE

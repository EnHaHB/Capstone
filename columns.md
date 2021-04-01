## Randomisation Data

Column | Description | Dtype
:--- | :--- | :---
HOSPNUM | Hospital number | int64
RDELAY | Delay between stroke and randomisation in hours | int64
RCONSC | Conscious state at randomisation (F - fully alert, D - drowsy, U - unconscious) | object
SEX | M = male; F = female | object
AGE | Age in years | int64
RSLEEP | Symptoms noted on waking (Y/N) | object
RATRIAL | Atrial fibrillation (Y/N); not coded for pilot phase - 484 patients (984). <br /> Atrial fibrillation is an abnormal heart rhythm (arrhythmia) characterized by the rapid and irregular beating of the atrial chambers of the heart. | object
RCT | CT before randomistion (Y/N). <br /> A CT scan or computed tomography scan is a medical imaging technique that uses computer-processed combinations of multiple X-ray measurements taken from different angles to produce tomographic (cross-sectional) images of a body. | object
RVISINF | Infarct visible on CT (Y/N) | object
RHEP24 | Heparin within 24 hours prior to randomisation (Y/N) | object
RASP3 | Aspirin within 3 days prior to randomisation (Y/N) | object
RSBP | Systolic blood pressure at randomisation (mmHg) | int64
RDEF1 | Face deficit (Y/N/C=can't assess) | object
RDFE2 | Arm/hand deficit (Y/N/C=can't assess) | object
RDEF3 | Leg/foot deficit (Y/N/C=can't assess) | object
RDEF4 | Dysphasia (Y/N/C=can't assess) <br /> Dysphasia is a partial or complete impairment of the ability to communicate resulting from brain injury. | object
RDEF5 | Hemianopia (Y/N/C=can't assess) <br /> Defective vision or blindness in half of the visual field; usually applied to bilateral defects caused by a single lesion. | object
RDEF6 | Visuospatial disorder (Y/N/C=can't assess) <br /> Pertaining to visual perception of spatial relationships. | object
RDEF7 | Brainstem/cerebellar signs (Y/N/C=can't assess) <br /> A cerebellar stroke is one of the less common types of strokes. It occurs when a blood vessel is blocked or bleeding, causing complete interruption to a portion of the cerebellum. | object
RDEF8 | Other deficit (Y/N/C=can't assess) | object
STYPE | Types of ischaemic strokes (more info in notebook) <br />TACS - Total anterior circulation syndrome  <br />PACS - Partial anterior circulation syndrome <br />POCS - Posterior circulation syndrome <br />LACS - Lacunar syndrome (LACS) <br />OTH - other | object
RDATE	| Year and month of randomisation (mm-YY) | object
HOURLOCAL	| Local time - hours (99-missing data) of randomisation | int64
MINLOCAL	| Local time - minutes (99-missing data) of randomisation | int64
DAYLOCAL	| Estimate of local day of week (1-Sunday, 2-Monday, 3-Tuesday, 4-Wednesday, 5-Thursday, 6-Friday, 7-Saturday) | int64
RXASP | Trial aspirin allocated (Y/N) | object
RXHEP | Trial heparin allocated (M/L/N) <br />The terminology for the allocated dose of unfractioned heparin changed slightly from the pilot to the main study. Patients were allocated either 12500 units subcutaneously twice daily (coded as H in the pilot and M in the main trial), 5000 units twice daily (coded as L throughout) or to 'avoid heparin' (coded as N throughout). | object


### Data collected on 14 day/discharge form about treatments given in hospital

Column | Description | Dtype
:--- | :--- | :---
DASP14 | Aspirin given for 14 days or till death or discharge (Y/N/U=unknown) | object
DASPLT | Discharged on long term aspirin (Y/N/U=unknown) | object
DLH14 | Low dose heparin given for 14 days or till death/discharge (Y/N/U=unknown) | object
DMH14 | Medium dose heparin given for 14 days or till death/discharge (Y/N/U=unknown) | object
DHH14 | Medium dose heparin given for 14 days etc in pilot (combine with above; Y/N) | object
ONDRUG | Estimate of time in days on trial treatment | object
DSCH | Non trial subcutaneous heparin (administered under the skin; Y/N/U=unknown) | object
DIVH | Non trial intravenous heparin (administered into vein, Y/N/U=unknown) | object
DAP | Non trial antiplatelet drug (Y/N/U=unknown) | object
DOAC | Other anticoagulants (Y/N/U=unknown) <br /> Members of a class of pharmaceuticals that decrease platelet aggregation and inhibit thrombus formation. | object
DGORM | Glycerol or manitol (Y/N/U=unknown) <br /> Both reduce intracranial pressure. | object
DSTER | Steroids (Y/N/U=unknown) | object
DCAA | Calcium antagonists (Y/N/U=unknown) | object
DHAEMD | Haemodilution (Y/N/U=unknown)  <br /> Increase in the volume of plasma in relation to red blood cells; reduced concentration of red blood cells in the circulation. | object
DCAREND | Carotid surgery (Y/N/U=unknown) <br /> Carotid artery stenosis is a narrowing or constriction of any part of the carotid arteries. | object
DTRHOMB | Thrombolysis (Y/N/U=unknown) <br />  Thrombolysis is the breakdown (lysis) of blood clots formed in blood vessels, using medication. | object
DMAJNCH | Major non-cerebral haemorrhage (Y/N/U=unknown) | object
DMAJNCHD | Date of above (days elapsed from randomisation) | float64
DMAJNCHX |Comment on above | object
DSIDE | Other side effect (Y/N/U=unknown) | object
DSIDED | Date of above (days elapsed from randomisation) | float64
DSIDEX |Comment on above | object

### Final diagnosis of initial event

Column | Description | Dtype
:--- | :--- | :---
DDIAGISC | Ischaemic stroke (Y/N/U=unknown) <br /> Ischaemic stroke is the most common kind of a stroke. Due to a blockage or narrowing in the arteries the supply of blood and oxygen to the brain is restricted causing an ischaemic stroke that can result in an infarction (necrotic tissue). See STPYE for different types. | object
DDIAGHA	| Haemorrhagic stroke (Y/N/U=unknown) <br /> Haemorrhagic stroke is a sudden bleeding into the tissues of the brain, into its ventricles, or into both. object
DDIAGUN	| Indeterminate stroke (Y/N/U=unknown) | object
DNOSTRK	| Not a stroke (Y/N/U=unknown) | object
DNOSTRKX | Comment on above | object

### Recurrent stroke within 14 days

Column | Description | Dtype
:--- | :--- | :---
DRSISC	| Ischaemic recurrent stroke (Y/N/U=unknown) | object
DRSISCD	| Date of above (days elapsed from randomisation) | float64
DRSH	| Haemorrhagic stroke (Y/N/U=unknown) | object
DRSHD	| Date of above (days elapsed from randomisation) | float64
DRSUNK	| Unknown type (Y/N/U=unknown) | object
DRSUNKD	| Date of above (days elapsed from randomisation) | float64

### Other events within 14 days

Column | Description | Dtype
:--- | :--- | :---
DPE	| Pulmonary embolism; (Y/N/U=unknown) <br /> Pulmonary embolism (PE) is a blockage of an artery in the lungs by a substance that has moved from elsewhere in the body through the bloodstream (embolism).| object
DPED | Date of above (days elapsed from randomisation) | object
DALIVE | Discharged alive from hospital (Y/N/U=unknown) | object
DALIVED	| Date of above (days elapsed from randomisation) | object
DPLACE	| Discharge destination <br /> A-Home <br /> B-Relatives home <br /> C-Residential care <br /> D-Nursing home <br /> E-Other hospital departments <br /> U-Unknown | object
DDEAD	| Dead on discharge form (Y/N/U=unknown) | object
DDEADD	| Date of above (days elapsed from randomisation); NOTE: this death is not necessarily within 14 days of randomisation | float64
DDEADC	| Cause of death <br /> 1-Initial stroke <br /> 2-Recurrent stroke (ischaemic or unknown)<br /> 3-Recurrent stroke (haemorrhagic) <br /> 4-Pneumonia <br /> 5-Coronary heart disease <br /> 6-Pulmonary embolism <br /> 7-Other vascular or unknown <br /> 8-Non-vascular <br /> 0-unknown | float64

### Data collected at 6 months

Column | Description | Dtype
:--- | :--- | :---
FDEAD	| Dead at six month follow-up (Y/N/U=unknown) | object
FLASTD	| Date of last contact (days elapsed from randomisation) | float64
FDEADD	| Date of death (days elapsed from randomisation); NOTE: this death is not necessarily within 6 months of randomisation| float64
FDEADC	| Cause of death <br /> 1-Initial stroke <br /> 2-Recurrent stroke (ischaemic or unknown)<br /> 3-Recurrent stroke (haemorrhagic) <br /> 4-Pneumonia <br /> 5-Coronary heart disease <br /> 6-Pulmonary embolism <br /> 7-Other vascular or unknown <br /> 8-Non-vascular <br /> 0-unknown | float64
FDEADX	| Comment on death | object
FRECOVER	| Fully recovered at 6 month follow-up (Y/N/U=unknown) | object
FDENNIS	| Dependent at 6 month follow-up (Y/N/U=unknown) | object
FPLACE	| Place of residance at 6 month follow-up <br /> A-Home <br /> B-Relatives home <br /> C-Residential care <br /> D-Nursing home <br /> E-Other hospital departments <br /> U-Unknown | object
FAP	| On antiplatelet drugs at six month follow-up (Y/N/U=unknown) | object
FOAC	| On oral anticoagulants at six month follow-up (Y/N/U=unknown) | object

### Other data and derived variables

Column | Description | Dtype
:--- | :--- | :---
FU1_RECD| Date discharge form received (days elapsed from randomisation) | float64
FU2_DONE| Date 6 month follow-up done (days elapsed from randomisation) | float64
COUNTRY| Abbreviated country code | object
CNTRYNUM| Country code (see Table 1) | int64
FU1_COMP| Date discharge form completed (days elapsed from randomisation)  | float64
NCCODE| Coding of compliance (see Table 3) | object
CMPLASP	| Compliant for aspirin (N/Y) | object
CMPLHEP	| Compliant for heparin (N/Y) | object
ID	| Indicator variable for death (1 = died; 0 = did not die)
TD	| Time of death or censoring in days  | float64
EXPDD	| Predicted probability of death/dependence at 6 month  | float64
EXPD6	| Predicted probability of death at 6 month  | float64
EXPD14	| Predicted probability of death at 14 days  | float64
SET14D	| Know to be dead or alive at 14 days (1 = Yes, 0 = No); this does not necessarily mean that we know outcome at 6 months - see OCCODE for this.  | int64
ID14	| Indicator of death at 14 days (1 = Yes, 0 = No)  | int64
OCCODE	| Six month outcome <br />1-dead <br />2-dependent <br />3-not recovered <br />4-recovered <br />0 or 9 - missing status  | int64

### Indicator variables for specific causes of death

Column | Description | Dtype
:--- | :--- | :---
DEAD1	| Initial stroke (1 = Yes, 0 = No) | int64
DEAD2	| Reccurent ischaemic/unknown stroke (1 = Yes, 0 = No) | int64
DEAD3	| Reccurent haemorrhagic stroke (1 = Yes, 0 = No) | int64
DEAD4	| Pneumonia (1 = Yes, 0 = No) | int64
DEAD5	| Coronary heart disease (1 = Yes, 0 = No) | int64
DEAD6	| Pulmonary embolism (1 = Yes, 0 = No) | int64
DEAD7	| Other vascular or unknown (1 = Yes, 0 = No) | int64
DEAD8	| Non vascular (1 = Yes, 0 = No) | int64
H14	| Cerebral bleed/heamorrhagic stroke within 14 days; this is slightly wider definition than DRSH and is used for analysis of cerebral bleeds; (1 = Yes, 0 = No) | int64
ISC14	| Indicator of ischaemic stroke within 14 days (1 = Yes, 0 = No) | int64
NK14	| Indicator of indeterminate stroke within 14 days (1 = Yes, 0 = No) | int64
STRK14	| Indicator of any stroke within 14 days (1 = Yes, 0 = No) | int64
HTI14	| Indicator of haemorrhagic transformation within 14 days (1 = Yes, 0 = No) | int64
PE14	| Indicator of pulmonary embolism within 14 days (1 = Yes, 0 = No) | int64
DVT14	| Indicator of deep vein thrombosis on discharge form (1 = Yes, 0 = No) | int64
TRAN14	| Indicator of major non-cerebral bleed within 14 days (1 = Yes, 0 = No) | int64
NCB14	| Indicator of any non-cerebral bleed within 14 days (1 = Yes, 0 = No) | int64
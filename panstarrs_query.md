
```sql
SELECT o.objID, 
 o.raStack, o.decStack, o.raStackErr, o.decStackErr,
 o.raMean, o.decMean, o.raMeanErr, o.decMeanErr, 
 o.nStackDetections, o.nDetections,
 o.qualityFlag,
 m.gFApFlux, m.gFApFluxErr, m.gFlags,
 m.rFApFlux, m.rFApFluxErr, m.rFlags,
 m.iFApFlux, m.iFApFluxErr, m.iFlags,
 m.zFApFlux, m.zFApFluxErr, m.zFlags,
 m.yFApFlux, m.yFApFluxErr, m.yFlags,
 s.gPSFMag, s.gPSFMagErr, 
 s.rPSFMag, s.rPSFMagErr, 
 s.iPSFMag, s.iPSFMagErr,
 s.zPSFMag, s.zPSFMagErr, 
 s.yPSFMag, s.yPSFMagErr
INTO mydb.hetdex_full_1
FROM ObjectThin o
 INNER JOIN StackObjectThin s ON o.objID = s.objID
 FULL OUTER JOIN ForcedMeanObject m ON o.objID = m.objID
WHERE
     o.raMean BETWEEN 160.0 AND 232.0
 AND o.decMean BETWEEN 42. AND 62.
 AND o.nStackDetections > 0
 AND o.nDetections > 0 
```
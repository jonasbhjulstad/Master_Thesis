      SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, NCALCF, ITYPEE, 
     *                   ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, 
     *                   ICALCF, LTYPEE, LSTAEV, LELVAR, LNTVAR, 
     *                   LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, 
     *                   LEPVLU, IFFLAG, IFSTAT )
      INTEGER NCALCF, IFFLAG, LTYPEE, LSTAEV, LELVAR, LNTVAR
      INTEGER LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, LEPVLU
      INTEGER IFSTAT
      INTEGER ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
      INTEGER INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
      INTEGER ICALCF(LCALCF)
      DOUBLE PRECISION FUVALS(LFVALU), XVALUE(LXVALU), EPVALU(LEPVLU)
C
C  Problem name : HS101     
C
C  -- produced by SIFdecode 1.0
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , V4    , V5    
      DOUBLE PRECISION V6    , P1    , P2    , P3    , P4    
      DOUBLE PRECISION P5    , P6    , FVALUE
      IFSTAT = 0
      DO     5 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4
     *                                                        ), IELTYP
C
C  Element type : 3PR       
C
    1  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       P1     = EPVALU(IPSTRT+     1)
       P2     = EPVALU(IPSTRT+     2)
       P3     = EPVALU(IPSTRT+     3)
       FVALUE = (V1 ** P1)*(V2 ** P2)*(V3 ** P3)         
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= FVALUE                                   
       ELSE
        FUVALS(IGSTRT+     1)= FVALUE * (P1 / V1)                       
        FUVALS(IGSTRT+     2)= FVALUE * (P2 / V2)                       
        FUVALS(IGSTRT+     3)= FVALUE * (P3 / V3)                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=FVALUE * (P1 / V1) * ((P1-1.0) / V1)     
         FUVALS(IHSTRT+     3)=FVALUE * (P2 / V2) * ((P2-1.0) / V2)     
         FUVALS(IHSTRT+     6)=FVALUE * (P3 / V3) * ((P3-1.0) / V3)     
         FUVALS(IHSTRT+     2)=FVALUE * (P1 / V1) * (P2 / V2)           
         FUVALS(IHSTRT+     4)=FVALUE * (P1 / V1) * (P3 / V3)           
         FUVALS(IHSTRT+     5)=FVALUE * (P2 / V2) * (P3 / V3)           
        END IF
       END IF
       GO TO     5
C
C  Element type : 4PR       
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       V4     = XVALUE(IELVAR(ILSTRT+     4))
       P1     = EPVALU(IPSTRT+     1)
       P2     = EPVALU(IPSTRT+     2)
       P3     = EPVALU(IPSTRT+     3)
       P4     = EPVALU(IPSTRT+     4)
       FVALUE = (V1 ** P1)*(V2 ** P2)*(V3 ** P3)         
     *           *(V4 ** P4)                             
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= FVALUE                                   
       ELSE
        FUVALS(IGSTRT+     1)= FVALUE * (P1 / V1)                       
        FUVALS(IGSTRT+     2)= FVALUE * (P2 / V2)                       
        FUVALS(IGSTRT+     3)= FVALUE * (P3 / V3)                       
        FUVALS(IGSTRT+     4)= FVALUE * (P4 / V4)                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=FVALUE * (P1 / V1) * ((P1-1.0) / V1)     
         FUVALS(IHSTRT+     3)=FVALUE * (P2 / V2) * ((P2-1.0) / V2)     
         FUVALS(IHSTRT+     6)=FVALUE * (P3 / V3) * ((P3-1.0) / V3)     
         FUVALS(IHSTRT+    10)=FVALUE * (P4 / V4) * ((P4-1.0) / V4)     
         FUVALS(IHSTRT+     2)=FVALUE * (P1 / V1) * (P2 / V2)           
         FUVALS(IHSTRT+     4)=FVALUE * (P1 / V1) * (P3 / V3)           
         FUVALS(IHSTRT+     7)=FVALUE * (P1 / V1) * (P4 / V4)           
         FUVALS(IHSTRT+     5)=FVALUE * (P2 / V2) * (P3 / V3)           
         FUVALS(IHSTRT+     8)=FVALUE * (P2 / V2) * (P4 / V4)           
         FUVALS(IHSTRT+     9)=FVALUE * (P3 / V3) * (P4 / V4)           
        END IF
       END IF
       GO TO     5
C
C  Element type : 5PR       
C
    3  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       V4     = XVALUE(IELVAR(ILSTRT+     4))
       V5     = XVALUE(IELVAR(ILSTRT+     5))
       P1     = EPVALU(IPSTRT+     1)
       P2     = EPVALU(IPSTRT+     2)
       P3     = EPVALU(IPSTRT+     3)
       P4     = EPVALU(IPSTRT+     4)
       P5     = EPVALU(IPSTRT+     5)
       FVALUE = (V1 ** P1)*(V2 ** P2)*(V3 ** P3)         
     *           *(V4 ** P4)*(V5 ** P5)                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= FVALUE                                   
       ELSE
        FUVALS(IGSTRT+     1)= FVALUE * (P1 / V1)                       
        FUVALS(IGSTRT+     2)= FVALUE * (P2 / V2)                       
        FUVALS(IGSTRT+     3)= FVALUE * (P3 / V3)                       
        FUVALS(IGSTRT+     4)= FVALUE * (P4 / V4)                       
        FUVALS(IGSTRT+     5)= FVALUE * (P5 / V5)                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=FVALUE * (P1 / V1) * ((P1-1.0) / V1)     
         FUVALS(IHSTRT+     3)=FVALUE * (P2 / V2) * ((P2-1.0) / V2)     
         FUVALS(IHSTRT+     6)=FVALUE * (P3 / V3) * ((P3-1.0) / V3)     
         FUVALS(IHSTRT+    10)=FVALUE * (P4 / V4) * ((P4-1.0) / V4)     
         FUVALS(IHSTRT+    15)=FVALUE * (P5 / V5) * ((P5-1.0) / V5)     
         FUVALS(IHSTRT+     2)=FVALUE * (P1 / V1) * (P2 / V2)           
         FUVALS(IHSTRT+     4)=FVALUE * (P1 / V1) * (P3 / V3)           
         FUVALS(IHSTRT+     7)=FVALUE * (P1 / V1) * (P4 / V4)           
         FUVALS(IHSTRT+    11)=FVALUE * (P1 / V1) * (P5 / V5)           
         FUVALS(IHSTRT+     5)=FVALUE * (P2 / V2) * (P3 / V3)           
         FUVALS(IHSTRT+     8)=FVALUE * (P2 / V2) * (P4 / V4)           
         FUVALS(IHSTRT+    12)=FVALUE * (P2 / V2) * (P5 / V5)           
         FUVALS(IHSTRT+     9)=FVALUE * (P3 / V3) * (P4 / V4)           
         FUVALS(IHSTRT+    13)=FVALUE * (P3 / V3) * (P5 / V5)           
         FUVALS(IHSTRT+    14)=FVALUE * (P4 / V4) * (P5 / V5)           
        END IF
       END IF
       GO TO     5
C
C  Element type : 6PR       
C
    4  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       V4     = XVALUE(IELVAR(ILSTRT+     4))
       V5     = XVALUE(IELVAR(ILSTRT+     5))
       V6     = XVALUE(IELVAR(ILSTRT+     6))
       P1     = EPVALU(IPSTRT+     1)
       P6     = EPVALU(IPSTRT+     2)
       P2     = EPVALU(IPSTRT+     3)
       P3     = EPVALU(IPSTRT+     4)
       P4     = EPVALU(IPSTRT+     5)
       P5     = EPVALU(IPSTRT+     6)
       FVALUE = (V1 ** P1)*(V2 ** P2)*(V3 ** P3)         
     *           *(V4 ** P4)*(V5 ** P5)*(V6 ** P6)       
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= FVALUE                                   
       ELSE
        FUVALS(IGSTRT+     1)= FVALUE * (P1 / V1)                       
        FUVALS(IGSTRT+     2)= FVALUE * (P2 / V2)                       
        FUVALS(IGSTRT+     3)= FVALUE * (P3 / V3)                       
        FUVALS(IGSTRT+     4)= FVALUE * (P4 / V4)                       
        FUVALS(IGSTRT+     5)= FVALUE * (P5 / V5)                       
        FUVALS(IGSTRT+     6)= FVALUE * (P6 / V6)                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=FVALUE * (P1 / V1) * ((P1-1.0) / V1)     
         FUVALS(IHSTRT+     3)=FVALUE * (P2 / V2) * ((P2-1.0) / V2)     
         FUVALS(IHSTRT+     6)=FVALUE * (P3 / V3) * ((P3-1.0) / V3)     
         FUVALS(IHSTRT+    10)=FVALUE * (P4 / V4) * ((P4-1.0) / V4)     
         FUVALS(IHSTRT+    15)=FVALUE * (P5 / V5) * ((P5-1.0) / V5)     
         FUVALS(IHSTRT+    21)=FVALUE * (P6 / V6) * ((P6-1.0) / V6)     
         FUVALS(IHSTRT+     2)=FVALUE * (P1 / V1) * (P2 / V2)           
         FUVALS(IHSTRT+     4)=FVALUE * (P1 / V1) * (P3 / V3)           
         FUVALS(IHSTRT+     7)=FVALUE * (P1 / V1) * (P4 / V4)           
         FUVALS(IHSTRT+    11)=FVALUE * (P1 / V1) * (P5 / V5)           
         FUVALS(IHSTRT+    16)=FVALUE * (P1 / V1) * (P6 / V6)           
         FUVALS(IHSTRT+     5)=FVALUE * (P2 / V2) * (P3 / V3)           
         FUVALS(IHSTRT+     8)=FVALUE * (P2 / V2) * (P4 / V4)           
         FUVALS(IHSTRT+    12)=FVALUE * (P2 / V2) * (P5 / V5)           
         FUVALS(IHSTRT+    17)=FVALUE * (P2 / V2) * (P6 / V6)           
         FUVALS(IHSTRT+     9)=FVALUE * (P3 / V3) * (P4 / V4)           
         FUVALS(IHSTRT+    13)=FVALUE * (P3 / V3) * (P5 / V5)           
         FUVALS(IHSTRT+    18)=FVALUE * (P3 / V3) * (P6 / V6)           
         FUVALS(IHSTRT+    14)=FVALUE * (P4 / V4) * (P5 / V5)           
         FUVALS(IHSTRT+    19)=FVALUE * (P4 / V4) * (P6 / V6)           
         FUVALS(IHSTRT+    20)=FVALUE * (P5 / V5) * (P6 / V6)           
        END IF
       END IF
    5 CONTINUE
      RETURN
      END

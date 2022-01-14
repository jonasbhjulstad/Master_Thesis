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
C  Problem name : HS9       
C
C  -- produced by SIFdecode 1.0
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , PI    
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : SNCS      
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       PI     = 4.0*ATAN(1.0)                            
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= SIN(PI*V1/12.0)*COS(PI*V2/16.0)          
       ELSE
        FUVALS(IGSTRT+     1)= COS(PI*V1/12.0)*COS(PI*V2/16.0)*PI/12.0  
        FUVALS(IGSTRT+     2)= -SIN(PI*V1/12.0)*SIN(PI*V2/16.0)*PI/16.0 
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=-SIN(PI*V1/12.0)*COS(PI*V2/16.0)*        
     *                         PI*PI/144.0                              
         FUVALS(IHSTRT+     3)=-SIN(PI*V1/12.0)*COS(PI*V2/16.0)*        
     *                         PI*PI/256.0                              
         FUVALS(IHSTRT+     2)=-COS(PI*V1/12.0)*SIN(PI*V2/16.0)*        
     *                         PI*PI/192.0                              
        END IF
       END IF
    2 CONTINUE
      RETURN
      END

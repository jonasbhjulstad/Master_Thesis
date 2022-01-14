      DOUBLE PRECISION FUNCTION EVAL92( X1, X2, X3, X4, X5, X6,         
     *                                  G, H )                          
      DOUBLE PRECISION X1, X2, X3, X4, X5, X6                           
      DOUBLE PRECISION G( 6 ), H( 6, 6 )                                
      INTEGER          I , J , K , N , NP1, L                           
      PARAMETER        ( N = 6, NP1 = N + 1 )                           
C     LOGICAL          CALC                                             
      DOUBLE PRECISION ALPHA , MUJ2  , T, MUI, MUJ   , SMUI , U, SI ,   
     *                 RIJ   , RHOI  , RHOJ  , EU, AI, AIMUI2, CMUI     
      DOUBLE PRECISION X  ( N  ), P( NP1 ),                             
     *                 MU ( 30 ), A( 30 ), R( 30, 30 ), S( 30 ),        
     *                 RHO( 30 ), DRHO( 30, N ), D2RHO( 30, N, N )      
      INTRINSIC        SIN   , COS   , EXP                              
C     SAVE             A, AI, AIMUI2, CMUI, MUI, MUJ, R, S, SMUI, T     
C                                                                       
C  Set data.                                                            
C                                                                       
C     DATA CALC / .TRUE. /                                              
      DATA MU / 8.6033358901938017D-01,  3.4256184594817283D+00,        
     *          6.4372981791719468D+00,  9.5293344053619631D+00,        
     *          1.2645287223856643D+01,  1.5771284874815882D+01,        
     *          1.8902409956860023D+01,  2.2036496727938566D+01,        
     *          2.5172446326646664D+01,  2.8309642854452012D+01,        
     *          3.1447714637546234D+01,  3.4586424215288922D+01,        
     *          3.7725612827776501D+01,  4.0865170330488070D+01,        
     *          4.4005017920830845D+01,  4.7145097736761031D+01,        
     *          5.0285366337773652D+01,  5.3425790477394663D+01,        
     *          5.6566344279821521D+01,  5.9707007305335459D+01,        
     *          6.2847763194454451D+01,  6.5988598698490392D+01,        
     *          6.9129502973895256D+01,  7.2270467060308960D+01,        
     *          7.5411483488848148D+01,  7.8552545984242926D+01,        
     *          8.1693649235601683D+01,  8.4834788718042290D+01,        
     *          8.7975960552493220D+01,  9.1117161394464745D+01 /       
C                                                                       
C  Calculate integration constants.                                     
C                                                                       
C     IF ( CALC ) THEN                                                  
         T         = 2.0D+0 / 1.5D+1                                    
         DO 20 I   = 1, 30                                              
            MUI    = MU( I )                                            
            SMUI   = SIN( MUI )                                         
            CMUI   = COS( MUI )                                         
            AI     = 2.0D+0 * SMUI / ( MUI + SMUI * CMUI )              
            A( I ) = AI                                                 
            S( I ) = 2.0D+0 * AI * ( CMUI -  SMUI / MUI )               
            AIMUI2 = AI * MUI ** 2                                      
            DO 10 J = 1, I                                              
               IF ( I .NE. J ) THEN                                     
                  MUJ       = MU( J )                                   
                  R( I, J ) = 5.0D-1 * (                                
     *                        SIN( MUI + MUJ ) / ( MUI + MUJ ) +        
     *                        SIN( MUI - MUJ ) / ( MUI - MUJ ) ) *      
     *                        AIMUI2 * A( J ) * MUJ ** 2                
                  R( J, I ) = R( I, J )                                 
               ELSE                                                     
                  R( I, I ) = 5.0D-1 * ( 1.0D+0 + 5.0D-1 *              
     *                        SIN( MUI + MUI ) / MUI ) *                
     *                        AIMUI2 ** 2                               
               END IF                                                   
   10       CONTINUE                                                    
   20    CONTINUE                                                       
C        CALC = .FALSE.                                                 
C     END IF                                                            
C                                                                       
C  Assign values to variables.                                          
C                                                                       
      X( 1 ) = X1                                                       
      X( 2 ) = X2                                                       
      X( 3 ) = X3                                                       
      X( 4 ) = X4                                                       
      X( 5 ) = X5                                                       
      X( 6 ) = X6                                                       
C                                                                       
C                                  n   2                                
C  Calculate the functions p(x) = SUM x .                               
C                           j     i=j  i                                
C                                                                       
      P( NP1 ) = 0.0D+0                                                 
      DO 100 K = N, 1, - 1                                              
         P( K ) = P( K + 1 ) + X( K ) ** 2                              
  100 CONTINUE                                                          
C                                                                       
C  Calculate the functions rho.                                         
C                                                                       
      DO 190 J = 1, 30                                                  
         MUJ2  = MU( J ) * MU( J )                                      
         U     = EXP( - MUJ2 * P( 1 ) )                                 
         DO 120 K = 1, N                                                
            DRHO( J, K ) = 2.0D+0 * U * X( K )                          
            DO 110 L = K, N                                             
              D2RHO( J, K, L ) = - 4.0D+0 * MUJ2 * U * X( K ) * X( L )  
              IF ( L .EQ. K ) D2RHO( J, K, L ) = D2RHO( J, K, L ) +     
     *                                           2.0D+0 * U             
  110       CONTINUE                                                    
  120    CONTINUE                                                       
         ALPHA = - 2.0D+0                                               
         DO 180 I = 2, N                                                
            EU = ALPHA * EXP( - MUJ2 * P( I ) )                         
            U = U + EU                                                  
            DO 140 K = I, N                                             
               DRHO( J, K ) = DRHO( J, K ) + 2.0D+0 * EU * X( K )       
               DO 130 L = K, N                                          
                  D2RHO( J, K, L ) = D2RHO( J, K, L ) -                 
     *               4.0D+0 * MUJ2 * EU * X( K ) * X( L )               
                  IF ( L .EQ. K )                                       
     *               D2RHO( J, K, L ) = D2RHO( J, K, L ) + 2.0D+0 * EU  
  130          CONTINUE                                                 
  140       CONTINUE                                                    
            ALPHA = - ALPHA                                             
  180    CONTINUE                                                       
         U = U + 5.0D-1 * ALPHA                                         
         RHO( J ) = - U / MUJ2                                          
  190 CONTINUE                                                          
C                                                                       
C  Evaluate the function and derivatives.                               
C                                                                       
      EVAL92    = T                                                     
      DO 320 K = 1, N                                                   
         G( K ) = 0.0D+0                                                
         DO 310 L = K, N                                                
           H( K, L ) = 0.0D+0                                           
  310    CONTINUE                                                       
  320 CONTINUE                                                          
      DO 490 I  = 1, 30                                                 
         SI     = S( I )                                                
         RHOI   = RHO( I )                                              
         EVAL92 = EVAL92 + SI * RHOI                                    
         DO 420 K = 1, N                                                
            G( K ) = G( K ) + SI * DRHO( I, K )                         
            DO 410 L = K, N                                             
               H( K, L ) = H( K, L ) + SI * D2RHO( I, K, L )            
  410       CONTINUE                                                    
  420    CONTINUE                                                       
         DO 480 J  = 1, 30                                              
            RIJ    = R( I, J )                                          
            RHOJ   = RHO( J )                                           
            EVAL92 = EVAL92 + RIJ * RHOI * RHOJ                         
            DO 440 K = 1, N                                             
               G( K ) = G( K ) + RIJ * ( RHOI * DRHO( J, K ) +          
     *                                   RHOJ * DRHO( I, K ) )          
               DO 430 L = K, N                                          
                 H( K, L ) = H( K, L ) + RIJ * (                        
     *                                 RHOI * D2RHO( J, K, L ) +        
     *                                 RHOJ * D2RHO( I, K, L ) +        
     *                                 DRHO( I, K ) * DRHO( J, L ) +    
     *                                 DRHO( J, K ) * DRHO( I, L ) )    
  430          CONTINUE                                                 
  440       CONTINUE                                                    
  480    CONTINUE                                                       
  490 CONTINUE                                                          
      RETURN                                                            
      END                                                               

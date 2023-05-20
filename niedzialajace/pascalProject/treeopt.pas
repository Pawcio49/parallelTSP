    (*                                         3-Opt TSP Algorithm
     
            INPUT   :      The associated datafile is "ThreeoptDatafile".
                             1st number represents # of nodes in teh given network.
                             2nd set of  numbers represents an NxN mweight matrix
                                 of the given  network.
                             3rd set of numbers represents the initial TS route.
                                 This may obtain from  the FITSP algorithm.
                                 (See fitsp.p program)
     
            Output   :       Outputs of  2-Opt TSP Algorithm are
                             1. The final route of TSP.
                             2. Total weight of the final route.
     
            Algorithm   :   The 3-Opt algorithm starts with the initial solution
                            specified by the array ROUTE in an undirected N-node
                            network given as the weigt matrix W.It exchanges three
                            edges at a time, and continues to do so until there
                            is no set of three edges in the current TS route whose
                            exchange would  yield a better route.
                            The initial TS route may obtain from the FITSP
                            algorithm.
     
            Note :          The choice of the route (Hamiltonian cycle) can have
                            the most dramatic impact on the final solution.
                            Therefore, a good initial route is preferred than
                               a random initial route.
                                                                    *)
     
    program Three_Opt_Approx (input,output,ThreeoptDatafile,ThreeoptOutfile);
     
    const	maxvar = 50;
     
    type	(* CHARFILE = file of char; *)
    	ARRN = array [1..maxvar] of integer;
    	ARRNN = array[1..maxvar,1..maxvar] of integer;
     
    var	N : integer;
    	W : ARRNN;
    	ROUTE : ARRN;
    	Nextint : integer;
    (*	ThreeoptDatafile : CHARFILE; *)
    (*	ThreeoptOutfile  : CHARFILE; *)
    	ThreeoptDatafile : TEXT;
    	ThreeoptOutfile  : TEXT;
     
     
    procedure Infile (var N : integer;
    		  var W : ARRNN;
    		  var ROUTE : ARRN;
    		  var Nextint : integer);
     
    var row, column : integer;
     
    begin
      reset(ThreeoptDatafile);
      readln(ThreeoptDatafile, Nextint);
      N := Nextint;
      for row := 1 to N do
      begin
        for column := 1 to N do
        begin
          read(ThreeoptDatafile,Nextint);
          W[row,column] := Nextint;
        end;
        readln(ThreeoptDatafile);
      end;
      for row := 1 to N do
      begin
        read(ThreeoptDatafile,Nextint);
        ROUTE[row] := Nextint;
      end;
      readln(ThreeoptDatafile);
    end;
     
     
    procedure THREEOPT(
           N      :integer;
       var W      :ARRNN;
       var ROUTE  :ARRN);
     
       type SWAPTYPE  =(ASYMMETRIC,SYMMETRIC);
            SWAPRECORD=record
                          X1,X2,Y1,Y2,Z1,Z2,GAIN:integer;
                          CHOICE                :SWAPTYPE
                       end;
       var BESTSWAP,SWAP:SWAPRECORD;
           I,INDEX,J,K  :integer;
           PTR          :ARRN;
     
       procedure SWAPCHECK(var SWAP:SWAPRECORD);
          var DELWEIGHT,MAX:integer;
       begin
          with SWAP do begin
             GAIN:=0;
             DELWEIGHT:=W[X1,X2]+W[Y1,Y2]+W[Z1,Z2];
             MAX:=DELWEIGHT-(W[Y1,X1]+W[Z1,X2]+W[Z2,Y2]);
             if MAX > GAIN then begin
                GAIN:=MAX;  CHOICE:=ASYMMETRIC
             end;
             MAX:=DELWEIGHT-(W[X1,Y2]+W[Z1,X2]+W[Y1,Z2]);
             if MAX > GAIN then begin
                GAIN:=MAX;  CHOICE :=SYMMETRIC
             end
          end  { with SWAP }
       end;  { SWAPCHECK }
     
       procedure REVERSE(START,FINISH:integer);
          var AHEAD,LAST,NEXT:integer;
       begin
          if START <> FINISH then begin
             LAST:=START;  NEXT:=PTR[LAST];
             repeat
                AHEAD:=PTR[NEXT];  PTR[NEXT]:=LAST;
                LAST:=NEXT;  NEXT:=AHEAD;
             until LAST = FINISH
          end  { if START <> FINISH }
       end;  { RESERVE }
     
    begin                                                   { MAIN BODY }
       for I:=1 to N-1 do PTR[ROUTE[I]]:=ROUTE[I+1];
       PTR[ROUTE[N]]:=ROUTE[1];
       repeat  { until BESTSWAP.GAIN = 0 }
          BESTSWAP.GAIN:=0;
          SWAP.X1:=1;
          for I:=1 to N do begin
             SWAP.X2:=PTR[SWAP.X1];  SWAP.Y1:=SWAP.X2;
             for J:=2 to N-3 do begin
                SWAP.Y2:=PTR[SWAP.Y1];  SWAP.Z1:=PTR[SWAP.Y2];
                for K:=J+2 to N-1 do begin
                   SWAP.Z2:=PTR[SWAP.Z1];
                   SWAPCHECK(SWAP);
                   if SWAP.GAIN > BESTSWAP.GAIN then BESTSWAP:=SWAP;
                   SWAP.Z1:=SWAP.Z2
                end;  { for K }
                SWAP.Y1:=SWAP.Y2
             end;  { for J }
             SWAP.X1:=SWAP.X2
          end;  { for I }
          if BESTSWAP.GAIN > 0 then
             with BESTSWAP do begin
                if CHOICE = ASYMMETRIC then begin
                   REVERSE(Z2,X1);
                   PTR[Y1]:=X1;  PTR[Z2]:=Y2
                end
                else begin
                   PTR[X1]:=Y2;  PTR[Y1]:=Z2
                end;
                PTR[Z1]:=X2;
             end  { with BESTSWAP }
       until BESTSWAP.GAIN = 0;
       INDEX:=1;
       for I:=1 to N do begin
          ROUTE[I]:=INDEX;  INDEX:=PTR[INDEX]
       end
    end;  { THREEOPT }
     
     
    procedure Outfile (ROUTE : ARRN);
    var count : integer;
    begin
      rewrite(ThreeoptOutfile);
      writeln (ThreeoptOutfile,'The route for tsp using Three_Opt_Approx is path');
      for count := 1 to N do
      begin
        write (ThreeoptOutfile,ROUTE[count]:5);
      end;
      writeln(ThreeoptOutfile);
    end;
     
    begin (* main *)
      Assign(ThreeoptDatafile,'threeopt.dat');
      Assign(ThreeoptOutfile,'threeopt.out');
        Infile (N,W,ROUTE,Nextint);
        THREEOPT (N,W,ROUTE);
        Outfile(ROUTE);
      Close(ThreeoptDatafile);
      Close(ThreeoptOutfile);
    end.



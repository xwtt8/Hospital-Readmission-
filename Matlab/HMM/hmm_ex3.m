% hidden markov model example from Matlab
% dishonest casino

states = [1,2]; %1= fair, 2 = loaded
symbols = [1,2]; %1 = heads, 2 = tails
p = [1,0]'; % model always starts from state 1
a = [0.95,0.05;0.1,0.9];
b = [0.5,0.5;0.1,0.9];

% hmmestimate (given observed and hidden state sequences)
T = 1000;
[oseq,hseq] = hmmgenerate(T,a,b);
[a_est,b_est] = hmmestimate(oseq,hseq);

% hmmtrain (EM method, given 100 observed sequences)
oseqs = {};

for i = 1:100
   T = randi([100,1000],1);
  [oseq,hseq] = hmmgenerate(T,a,b);
  oseqs{i} = oseq;
  oseq
end

r1 = rand; r2 = rand;
aguess = [r1,1-r1; r2,1-r2];
bguess = [0.5,0.5; 0.5,0.5];
[a_est1,b_est1] = hmmtrain(oseqs,aguess,bguess);
a_est1
b_est1


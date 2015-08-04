% using viterbi algorithm to get the most likely sequence

%% Initialization
clear; close all; clc

%% Initial tran matrix and emiss matrix
X = csvread('lab_test/kidney.csv');
numperrow = csvread('lab_test/kidney_number.csv');

a = [0.0205158357134212,0.979484164286579;0.764643595827838,0.235356404172163];
b = [0.223092442494099,0.776907557505902;0.406355745908130,0.593644254091870];

% convert all the 0 into 1 
X_=X;
X_(X_==0)=1;

%% Compute the most likely sequence
states = zeros(size(X));

for i = 1:size(X,1)
    tmp = X_(i,:);
    states(i,:) = hmmviterbi(tmp,a,b);
end

csvwrite('kidney_state.csv',states);
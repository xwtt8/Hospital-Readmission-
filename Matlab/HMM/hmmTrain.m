%Hmm trainer
%% Initialization
clear; close all; clc
%% Load Observations Data

X = csvread('lab_test/kidney.csv');
numperrow = csvread('lab_test/kidney_number.csv');

%% Train Hidden Markov Model
oseqs = {};
for i = 1:size(X,1)
    tmp = X(i,:);
    oseqs{i} = tmp(1:numperrow(i));
end

r1 = rand; r2 = rand;
aguess = [r1,1-r1; r2,1-r2];
bguess = [0.5,0.5; 0.5,0.5];
[a_est1,b_est1] = hmmtrain(oseqs,aguess,bguess,'maxiterations',500);



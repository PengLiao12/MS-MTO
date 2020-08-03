function data_MFEA = MFEA(Tasks,pop,gen,rmp,sub_pop,mu,mum)
%MFEA function: implementation of MFEA algorithm
%Gupta,A.,Ong,Y.S.,Feng,L.,2015. Multifactorial evolution: toward evolutionary multitasking. IEEE Transactions on Evolutionary Computation20,343¨C357.
%     clc
%     tic
if mod(pop,2) ~= 0
    pop = pop + 1;
end
no_of_tasks=length(Tasks)-1;
if no_of_tasks <= 1
    error('At least 2 tasks required for MFEA');
end
D=zeros(1,no_of_tasks);
for i=1:no_of_tasks
    D(i)=Tasks(i).dims;
end
D_multitask=max(D);

%     fnceval_calls = zeros(1,reps);
%     calls_per_individual=zeros(1,pop);
%     EvBestFitness = zeros(no_of_tasks*reps,gen);    % best fitness found
%     TotalEvaluations=zeros(reps,gen);               % total number of task evaluations so fer
bestobj=Inf(1,no_of_tasks);


length_sub_pop=length(sub_pop(:,1));
if length_sub_pop>pop
    length_sub_pop=pop;
else
    lhs = lhsdesign(pop-length_sub_pop,Tasks(1).dims);
end
for i = 1 : length_sub_pop
    population(i) = Chromosome();
    population(i).rnvec = sub_pop(i,1:Tasks(1).dims);
    population(i).skill_factor=0;
end
for i = 1 : pop-length_sub_pop
    population(i+length_sub_pop) = Chromosome();
    population(i+length_sub_pop).rnvec = lhs(i,1:Tasks(1).dims);
    population(i+length_sub_pop).skill_factor=0;
end
for i = 1 : pop
    [population(i),calls_per_individual(i)] = evaluate(population(i),Tasks,no_of_tasks);
end

% fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
% TotalEvaluations(rep,1)=fnceval_calls(rep);

factorial_cost=zeros(1,pop);
for i = 1:no_of_tasks
    for j = 1:pop
        factorial_cost(j)=population(j).factorial_costs(i);
    end
    [xxx,y]=sort(factorial_cost);
    population=population(y);
    for j=1:pop
        population(j).factorial_ranks(i)=j;
    end
    bestobj(i)=population(1).factorial_costs(i);
%     EvBestFitness(i+2*(rep-1),1)=bestobj(i);
    bestInd_data(1,i)=population(1);
end
for i=1:pop
    [xxx,yyy]=min(population(i).factorial_ranks);
    x=find(population(i).factorial_ranks == xxx);
    equivalent_skills=length(x);
    if equivalent_skills>1
        population(i).skill_factor=x(1+round((equivalent_skills-1)*rand(1)));
        tmp=population(i).factorial_costs(population(i).skill_factor);
        population(i).factorial_costs(1:no_of_tasks)=inf;
        population(i).factorial_costs(population(i).skill_factor)=tmp;
    else
        population(i).skill_factor=yyy;
        tmp=population(i).factorial_costs(population(i).skill_factor);
        population(i).factorial_costs(1:no_of_tasks)=inf;
        population(i).factorial_costs(population(i).skill_factor)=tmp;
    end
end

%         mu = 10;     % Index of Simulated Binary Crossover (tunable)
%         mum = 10;    % Index of polynomial mutation
generation=1;
while generation < gen
    generation = generation + 1;
    indorder = randperm(pop);
    count=1;
    for i = 1 : pop/2
        p1 = indorder(i);
        p2 = indorder(i+(pop/2));
        child(count)=Chromosome();
        child(count+1)=Chromosome();
        if (population(p1).skill_factor == population(p2).skill_factor) || (rand(1)<rmp)      % crossover
            u = rand(1,D_multitask);
            cf = zeros(1,D_multitask);
            cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
            cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
            child(count) = crossover(child(count),population(p1),population(p2),cf);
            child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
            if rand(1) < 1
                child(count)=mutate(child(count),child(count),D_multitask,mum);
                child(count+1)=mutate(child(count+1),child(count+1),D_multitask,mum);
            end
            sf1=1+round(rand(1));
            sf2=1+round(rand(1));
            if sf1 == 1 % skill factor selection
                child(count).skill_factor=population(p1).skill_factor;
            else
                child(count).skill_factor=population(p2).skill_factor;
            end
            
            if sf2 == 1
                child(count+1).skill_factor=population(p1).skill_factor;
            else
                child(count+1).skill_factor=population(p2).skill_factor;
            end
            
        else
            child(count)=mutate(child(count),population(p1),D_multitask,mum);
            child(count).skill_factor=population(p1).skill_factor;
            child(count+1)=mutate(child(count+1),population(p2),D_multitask,mum);
            child(count+1).skill_factor=population(p2).skill_factor;
        end
        count=count+2;
    end
    for i = 1 : pop
        [child(i),calls_per_individual(i)] = evaluate(child(i),Tasks,no_of_tasks);
    end
%     fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
%     TotalEvaluations(rep,generation)=fnceval_calls(rep);
    
    intpopulation(1:pop)=population;
    intpopulation(pop+1:2*pop)=child;
    factorial_cost=zeros(1,2*pop);
    for i = 1:no_of_tasks
        for j = 1:2*pop
            factorial_cost(j)=intpopulation(j).factorial_costs(i);
        end
        [xxx,y]=sort(factorial_cost);
        intpopulation=intpopulation(y);
        for j=1:2*pop
            intpopulation(j).factorial_ranks(i)=j;
        end
        if intpopulation(1).factorial_costs(i)<=bestobj(i)
            bestobj(i)=intpopulation(1).factorial_costs(i);
            bestInd_data(1,i)=intpopulation(1);
        end
%         EvBestFitness(i+2*(rep-1),generation)=bestobj(i);
    end
    for i=1:2*pop
        [xxx,yyy]=min(intpopulation(i).factorial_ranks);
        intpopulation(i).skill_factor=yyy;
        intpopulation(i).scalar_fitness=1/xxx;
    end
    
    
    [xxx,y]=sort(-[intpopulation.scalar_fitness]);
    intpopulation=intpopulation(y);
    population=intpopulation(1:pop);
    
    %             disp(['MFEA Generation = ', num2str(generation), ' best factorial costs = ', num2str(bestobj)]);
end
%     data_MFEA.wall_clock_time=toc;
%     data_MFEA.EvBestFitness=EvBestFitness;
data_MFEA.bestInd_data=bestInd_data;
%     data_MFEA.TotalEvaluations=TotalEvaluations;
end
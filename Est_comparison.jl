using Distributions, LinearAlgebra, GLM, Optim, Random


## initial configuration for simulation
B=500               #simulation number
N=1000               #sample size
T=2                  #two periods panel
dx=2                 #dimension of X
dz=2                 #dimension of Z


beta0=ones(dx,1)          #true coefficient for a single good
gamma0=ones(dz,1)         #true coefficent for incremental utility

eta1=0.5.*ones(dx,1)      #coefficient for fixed effects alpha=eta0+eta1*Xbar
eta0=0

betanon=ones(dx-1,B)       #nonparametric estimator for beta
gammanon=ones(dz-1,B)        #nonparametric estimator for gamma
err=ones(B)                #prob of having wrong signs of substitution patterns

betapar=ones(dx-1,B)       #parametric estimators
gammapar=ones(dz-1,B)
errpar=ones(B)

#two estimators assuming no bundles
betanon1=ones(dx-1,B)      #estimator under stationarity but assuming no bunlds
betafl=ones(dx-1,B)        #fixed effect logit estimator


#some functions
sg(y)=sign.(y)                     #determine the sign of y
D(x)=x[:,dx+1:end]-x[:,1:dx]       #difference of characteristics of two periods
dp(y)=y[:,2]-y[:,1]                #difference of choice probabilities

#G(x)=2 .*cdf.(Normal(0,1),abs.(x)).-1      #weight function for the value of Ys-Yt #2 .*cdf.(Normal(),abs.(x)).-1
G(x)=abs.(x)
G1(x)=x.*(x.>=0)                           #function for nonnegative value


# neural network setting--estimating conditional choice probability
alpha=10^(-3)                           #learn size
ite=200                                 #iteration number
phi(v)=1 ./(1 .+exp.(-v))               #activation function
dphi(v)=exp.(-v)./(1 .+exp.(-v)).^2     #derivative of phi

## function for neural network estimator for conditional choice probability
function neu(x,y)
#x:covariate;   y:outcome
dx=size(x,2)            #dimension of x
dy=size(y,2)            #dimension of y
n=size(y,1)             #length of data
Xt=[ones(n,1) x]        #add constant
w=ones(dx+1,dy)         #initial weights
for i=1:ite
   for j=1:n
    pre=phi(Xt[j,:]'*w)          #prediction by neural network
    err=y[j,:]'-pre              #prediction error
    w+=2*alpha.*Xt[j,:]*err.*dphi(Xt[j,:]'*w) #update  weight
   end
end
phat=phi(Xt*w)            #estimated probability
return phat
end

## criterion function in this paper
function mom(theta,XA,XB,Z,p)
    """
    #theta: true parameter
    #XA(XB): covariate for good A(B) at all perids
    #p: estimated choice probability for Y1, Y2, Y3, Y0
    """
    beta=[1;theta[1:dx-1]]
    gamma=[1;theta[dx:end]]
    Gamma=Z*gamma            #incremental utility

    deltaA=D(XA)*beta        #variation in covariate index for good A
    deltaB=D(XB)*beta        #variation in covariate index for good B

    p1=p[:,1:T]              #estimated chocie probability for choosing only good A
    p2=p[:,T+1:2*T]          #probability of choosing only good B
    p0=p[:,2*T+1:3*T]        #neither
    p3=p[:,3*T+1:4*T]        #two goods together
    p13=p1+p3                #demand for good A
    p23=p2+p3                #demand for good B

    ## ID1--construct index for a single good
    I1=1 .-((sg(dp(p1)).*deltaA.>0).|(sg(dp(p1)).*deltaB.<0))
    I2=1 .-((sg(dp(p2)).*deltaA.<0).|(sg(dp(p2)).*deltaB.>0))
    I3=1 .-((sg(dp(p3)).*deltaA.>0).|(sg(dp(p3)).*deltaB.>0))
    I0=1 .-((sg(dp(p0)).*deltaA.<0).|(sg(dp(p0)).*deltaB.<0))

    # construct objective function using a single choice
    Q1=mean(G1(dp(p1)).*I1+G1(dp(p2)).*I2+G1(dp(p3)).*I3+G1(dp(p0)).*I0)

    ## ID2--construct index for demand of good A and good B
    I13=1 .-((sg(dp(p13)).*deltaA.>0).|((sg(dp(p13)).*(deltaA+sg(Gamma).*deltaB).>0)))#.*(abs.(Gamma).>-sg(dp(p13)).*deltaA)))
    I23=1 .-((sg(dp(p23)).*deltaB.>0).|((sg(dp(p23)).*(deltaB+sg(Gamma).*deltaA).>0)))#.*(abs.(Gamma).>-sg(dp(p23)).*deltaB)))

    #objective function using conditional demand
    Q2=mean(G1(dp(p13)).*I13+G1(dp(p23)).*I23)

    ##ID3--construct index for sum of two choice probabilities
    Ib1(t1)=1 .-(Gamma.<min.((-1)^t1.*deltaA,(-1)^(t1-1).*deltaB)).*((-1)^t1.*(deltaA-deltaB).>0)
    Ib2(t1)=1 .-(Gamma.>max.((-1)^(t1-1).*deltaA,(-1)^(t1-1).*deltaB)).*((-1)^t1.*(deltaA+deltaB).>0)

    #objective function by sum of two choice probabilities
    L1(t1,t2)=p1[:,t1]+p2[:,t2].-1
    L2(t1,t2)=p3[:,t1]+p0[:,t2].-1
    Q3=mean(G1(L1(2,1)).*Ib1(2)+G1(L1(1,2)).*Ib1(1)+G1(L2(2,1)).*Ib2(2)+G1(L2(1,2)).*Ib2(1))

  return Q1+Q2+Q3
end


Random.seed!(1234)

## mixed-effect logit estimator: epj~Gumbel  alphaj=eta0+Xbar*eta1+vj where vj~N(0,1)
# use simulated method of moments to calculate choice probabilities
S=20                           #simulation numbers
ep0_s=rand(Gumbel(),N,T,S)     #simulated shocks
epA_s=rand(Gumbel(),N,T,S)
epB_s=rand(Gumbel(),N,T,S)
vA_s=randn(N,S)                #simulated error term in fixed effect
vB_s=randn(N,S)

muA_s=zeros(N,T,S)             #sum of two error terms: epA+vA
muB_s=zeros(N,T,S)
for t=1:T
  muA_s[:,t,:]=vA_s+epA_s[:,t,:]-ep0_s[:,t,:]
  muB_s[:,t,:]=vB_s+epB_s[:,t,:]-ep0_s[:,t,:]
end

#criterion function for the parametric method
function par_sim(theta,XA,XB,Z,Y1,Y2,Y3)
    beta=[1;theta[1:dx-1]]
    gamma=[1;theta[dx:dx+dz-2]]
    # beta=theta[1:dx]
    # gamma=theta[dx+1:dx+dz]
    eta1=theta[dx+dz-1:end-1]
    eta0=theta[end]

    alphaA=eta0.+((XA[:,1:dx]+XA[:,dx+1:end])/T)*eta1;   #mean fixed effect
    alphaB=eta0.+((XB[:,1:dx]+XB[:,dx+1:end])/T)*eta1;
    delA=[XA[:,1:dx]*beta XA[:,dx+1:end]*beta].+alphaA;  #mean utility
    delB=[XB[:,1:dx]*beta XB[:,dx+1:end]*beta].+alphaB;
    Gam=Z*gamma                                          #incremental utilty

    #simulated choice proability as a function of parameters
    p1_s=mean((muA_s.+delA.>=max.(muB_s.+delB,0)).*(muB_s.<=-delB.-Gam),dims=3)[:,:]
    p2_s=mean((muB_s.+delB.>=max.(muA_s.+delA,0)).*(muA_s.<=-delA.-Gam),dims=3)[:,:]
    p3_s=mean((muA_s.+delA.+Gam.>=max.(-muB_s.-delB,0)).*(muB_s.>=-delB.-Gam),dims=3)[:,:]

    X=[XA XB Z]'
    #moment conditions for the two periods
    Q_s1=[X*(Y1-p1_s)[:,1]; X*(Y2-p2_s)[:,1]; X*(Y3-p3_s)[:,1]]/N
    Q_s2=[X*(Y1-p1_s)[:,2]; X*(Y2-p2_s)[:,2]; X*(Y3-p3_s)[:,2]]/N

    Q=Q_s1'*Q_s1+Q_s2'*Q_s2
    return Q
end


## nonparmaetric estimator which does not allow bundles: gamma=-infty
function mom1(beta,XA,XB,p)
   b=[1;beta]

   deltaA=D(XA)*b               #variation in covariate index for good A
   deltaB=D(XB)*b               #variation in covariate index for good B

   p1=p[:,1:T]                  #estimated chocie probability for choosing only good A
   p2=p[:,2*T-1:2*T]            #only good A
   p0=p[:,3*T-1:3*T]            #neither

   ##  construct index for a single good
   I1=1 .-((sg(dp(p1)).*deltaA.>0).|(sg(dp(p1)).*(deltaA-deltaB).>0))
   I2=1 .-((sg(dp(p2)).*deltaB.>0).|(sg(dp(p2)).*(deltaB-deltaA).>0))
   I0=1 .-((sg(dp(p0)).*deltaA.<0).|(sg(dp(p0)).*deltaB.<0))

   # construct objective function using a single choice
   Q=mean(G(dp(p1)).*I1+G(dp(p2)).*I2+G(dp(p0)).*I0)
   return Q
end


## Chamberlain's fixed-effect logti model which does not allow bundles: gamma=-infty
function momlogit(beta,XA,XB,Y)
   b=[1;beta]
   dc(c1,c2)=(Y[:,1].==c1).*(Y[:,2].==c2)  #indicator for choosing c1 at t=1 and c2 at t=2
   p01=exp.(D(XA)*b)./(1 .+exp.(D(XA)*b)) #choice probability of choosing (0,1) conditional on y1+y2=1
   p02=exp.(D(XB)*b)./(1 .+exp.(D(XB)*b))
   p12=exp.((D(XB)-D(XA))*b)./(1 .+exp.((D(XB)-D(XA))*b))

   Q1=mean(dc(0,1).*log.(p01)+dc(1,0).*log.(1 .-p01))
   Q2=mean(dc(0,2).*log.(p02)+dc(2,0).*log.(1 .-p02))
   Q3=mean(dc(1,2).*log.(p12)+dc(2,1).*log.(1 .-p12))
   return -(Q1+Q2+Q3)
end


#Random.seed!(1234)

## simulation starts here
for b=1:B

   XA1=rand(MvNormal(zeros(dx),dx*I),N)'     #dx* XA=[XA1, XA2] XA1-characteristic at T=1
   XA2=rand(MvNormal(zeros(dx),dx*I),N)'
   XA=[XA1 XA2]
   XB1=rand(MvNormal(zeros(dx),dx*I),N)'
   XB2=rand(MvNormal(zeros(dx),dx*I),N)'
   XB=[XB1 XB2]
   Z=[rand(Normal(2,2),N) rand(Normal(0,1),N)]

   ## desidn 1-epj~Gumbel and independent over choices and time; alphaj~Xjbar*eta1+N(0,1)
   # epA=rand(Gumbel(),N,T)
   # epB=rand(Gumbel(),N,T)
   # ep0=rand(Gumbel(),N,T)
   # alphaA=(XA1+XA2)*beta0/4+randn(N,1)       #fixed effect for choice A
   # alphaB=(XB1+XB2)*beta0/4+randn(N,1)


   ## design 2: ep follow multivariate normal with correlation rho=-0.7; alphaj~Xjbar*eta1+N(0,1)
   # rho=-0.7
   # ep1=rand(MvNormal([2;-2],[1 rho;rho 1]),N)'
   # ep2=rand(MvNormal([2;-2],[1 rho;rho 1]),N)'
   # epA=[ep1[:,1] ep2[:,1]]
   # epB=[ep1[:,2] ep2[:,2]]
   # ep0=randn(N,T)
   # alphaA=(XA1+XA2)*beta0/4+randn(N,1)       #fixed effect for choice A
   # alphaB=(XB1+XB2)*beta0/4+randn(N,1)

   ## design 3:epj~Gumbel and independent over choices and time; alphaj=(Xjbar/2-Xkbar)*beta0*(1+2N(0,1))
   # epA=rand(Gumbel(),N,T)
   # epB=rand(Gumbel(),N,T)
   # ep0=rand(Gumbel(),N,T)
   # alphaA=((XA1+XA2-2*(XB1+XB2))*beta0/4).*(1 .+randn(N,1))
   # alphaB=((XB1+XB2-2*(XA1+XA2))*beta0/4).*(1 .+randn(N,1))

   ## design 4: ep follow multivariate normal with correlation rho=0.7; alphaj=(Xjbar/2-Xkbar)*beta0*(1+2N(0,1))
   rho=-0.7
   ep1=rand(MvNormal([2;-2],[1 rho;rho 1]),N)'
   ep2=rand(MvNormal([2;-2],[1 rho;rho 1]),N)'
   epA=[ep1[:,1] ep2[:,1]]
   epB=[ep1[:,2] ep2[:,2]]
   ep0=randn(N,T)
   alphaA=((XA1+XA2-2*(XB1+XB2))*beta0/4).*(1 .+randn(N,1))
   alphaB=((XB1+XB2-2*(XA1+XA2))*beta0/4).*(1 .+randn(N,1))


   ## generate utilities for all choices
   X=[XA XB Z]
   uA=[XA[:,1:dx]*beta0 XA[:,dx+1:end]*beta0].+alphaA+epA-ep0      #utility of only buy A
   uB=[XB[:,1:dx]*beta0 XB[:,dx+1:end]*beta0].+alphaB+epB-ep0      #utility of only buy B
   u0=0                                                            #utility of outside option
   uAB=uA+uB.+Z*gamma0                                             #utility of buy AB

   ## generate choices: 1-only buy A  2-only buy B  3-buy AB bundle  0-outside option
   um=max.(uA,uB,uAB,u0)
   Y=(um.==uA)+(um.==uB)*2+(um.==uAB)*3              #consumer choice
   Y1=(Y.==1)                                        #consumes who only buy A
   Y2=(Y.==2)                                        #consumes who only buy B
   Y3=(Y.==3)                                        #consumes who only buy AB
   Y0=(Y.==0)                                        #consumes who buy outside option


   ## nonparametric estimator in my paper
   phat=neu(X,[Y1 Y2 Y0 Y3])                #estimated choice probability
   obj(theta)=mom(theta,XA,XB,Z,phat)
   initial=[ones(dx-1);0.5 .*ones(dz-1)]
   opt=optimize(obj,initial,NelderMead())    #optimization with point identification
   betanon[:,b]=opt.minimizer[1:dx-1]
   gammanon[:,b]=opt.minimizer[dx:end]
   err[b]=mean(abs.((Z*gamma0.>=0)-(Z*[1;gammanon[:,b]].>=0)))


   ## parametric estimator for both beta and gamma
   objpar(theta)=par_sim(theta,XA,XB,Z,Y1,Y2,Y3)
   initial_par=[ones(dx+dz);0.5 .*ones(dx-1)]
   optpar=optimize(objpar,initial_par,NelderMead())
   betapar[:,b]=optpar.minimizer[1:dx-1]
   gammapar[:,b]=optpar.minimizer[dx:dz+dx-2]
   errpar[b]=mean(abs.((Z*gamma0.>=0)-(Z*[1;gammapar[:,b]].>=0)))


   ## Chamberlain's fixed-effect logit model--not allowing for bundles
   objlogit(beta)=momlogit(beta,XA,XB,Y)
   optfl=optimize(objlogit,-5,5)
   betafl[b]=optfl.minimizer

   ## nonparametric estimator for beta0 without bundle
   phat1=phat[:,1:3*T]
   obj1(beta)=mom1(beta,XA,XB,phat1)
   opt1=optimize(obj1,-5,5)
   betanon1[b]=opt1.minimizer


end


#evaluate the three estimators: bias,sd,MSE,MAD
function M(theta)
  return [mean(theta)-1 std(theta) sqrt(var(theta)+(mean(theta)-1)^2)  median(abs.(theta.-1))] #bias sd rMSE  MAD
end

## performance for beta
rMSEnon=M(betanon)
rMSEpar=M(betapar)
rMSEnon1=M(betanon1)
rMSEfl=M(betafl)

## performance for gamma
rMSEgnon=M(gammanon)
rMSEgpar=M(gammapar)
e=[mean(err) mean(errpar)]

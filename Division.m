function root=Division(fun,x,a,b)
p=-1;
while (fun(a)*fun(b) <=0) && (abs(a-b)>x)
    c=(a+b)/2;
    if fun(c)*fun(b)<=0
        a=c;
        p=p+1;
    else
        p=p+1;
        b=c;
    end
end
root = a;
end
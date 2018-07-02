%option noyywrap
%{
#include <stdio.h>
char str[10],e[30],s[3],o[5],cl[10],gpa[4],q,o1[30],branch[10],op[5];
int st,ro;
%}
 
%% 
done {printf("Input received..!!!\n");}
[%\t]+ ;          
a {
//show marks of a subject
printf("\nFor marks in subject type the following details : \n 'exam'\n");
q='a';
}

b {
printf("\nType the following details : \n 'avg/max/min' \n'exam' \n");
q='b';
}

c {
q='c';
}

d {
printf("\nstudents with back logs \n");
q='d';
}


e {
//printf("\nenter cgpa  'cg' / sgpa  'sg'\n");
q='e';
}


f {
q='f';
}


g {
q='g';
}


h {
q='h';
}


DS {
strcpy(s,"DS");
}

OS {
strcpy(s,"OS");
}

m1 {
strcpy(e,"m1");
}
mid {
strcpy(e,"mid");
}

m2 {
strcpy(e,"m2");
}

end {
strcpy(e,"end");
}
marks {
printf("you selected 'marks'\n enter choice\n");
strcpy(cl,"cse_marks");
}

grades {
printf("you selected 'grades'\nenter choice\n");
strcpy(cl,"cse_grades");
}

max {
strcpy(o,"max");
}

min {
strcpy(o,"min");
}

avg {
strcpy(o,"avg");
}
hru {
printf("I am fine..thank you\n what about you?");
}

fine {
printf("ok..let's start");
}


. {
return 0;
}

st {
st=1;
}

%%
void start()
{
yylex();
}




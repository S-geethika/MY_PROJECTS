#include <mysql.h>
#include <stdio.h>
#include <my_global.h>
int main() {
	MYSQL *conn;
	MYSQL_RES *res,*res1;
	MYSQL_ROW row,row1;
extern char str[10],e[30],o[5],cl[10],gpa[4],q,op[5],branch[10];
extern int st,ro;
float g;
char ex[10],c,sem,s[3],qq[90],sql1[100],sql[1200],ch,choice,o1[30];
int r,m,error=0,sm;

//connecting to data base
char *server = "localhost";
	char *user = "root";
	char *password = "root"; 
	char *database = "utp";
//count of fields to display
int disp;

conn = mysql_init(NULL);
if (!mysql_real_connect(conn, server, user, password, 
                                      database, 0, NULL, 0)) {
		fprintf(stderr, "%s\n", mysql_error(conn));
		exit(1);
	}
c='y';
while(c=='y')
{
	
printf("please select one of the below query domain\n1. For marks, type 'marks'\n2. For grades, type 'grades'\n  enter '.' after completion\n\n \n'marks' \n a.show marks of subject\n b.avg/min/max marks in subject \n c.students who got above 'm' marks \n d.students with backlogs \n \n 2. Grades Queries(type 'grades') \n e.semester grades\n f. toppers in each semester\n g. branch topper \n h.students above 'g' gpa\n");
strcpy(o1,"mg");
start();

if(strcmp(o1,"mg")==0)
{
switch(q)
{
case 'a':
{
disp=2;
printf("to know your marks type rollno,else type 0\n");
scanf("%d",&r);
printf("'subject'(if you want all sub,type 'all')\n");
scanf("%s",s);
if(r==0&&(strcmp(s,"all"))==0)//all rno,all sub
{
sprintf(sql, "select rollno,%s,sub from cse_marks",e);
disp=3;
}
else if(r==0&&(strcmp(s,"all"))!=0)//all rno,s
sprintf(sql, "select rollno,%s from cse_marks where sub='%s'",e,s);
else if(r!=0&&(strcmp(s,"all"))==0)//r,all sub
sprintf(sql, "select sub,%s from cse_marks where rollno=%d",e,r);
else//r,s
{
sprintf(sql, "select %s from cse_marks where rollno=%d and sub='%s'",e,r,s);
disp=1;
}
//break;
}
break;
case 'b'://avg/min/max marks in subject
{
printf("'subject'(if you want all sub,type 'all')\n");
scanf("%s",s);
if((strcmp(s,"all"))!=0)
{
sprintf(sql, "select %s(%s) from cse_marks where sub='%s' ",o,e,s);
disp=1;
}
else
{
sprintf(sql, "select sub,%s(%s) from cse_marks group by sub",o,e);
disp=2;
}
}
break;

case 'c'://students who got above 'm' marks
{
printf(" enter 'exam' :\n");
scanf("%s",&e);
printf(" m value :\n");
scanf("%d",&m);
disp=2;
if(m<0)
error=1;
else
sprintf(sql, "select distinct(rollno),%s from cse_marks where %s>%d",e,e,m);
}
break;

case 'd':
{
disp=1;
sprintf(sql, "select name from cse_marks where total<35");
}
break;

case 'e'://semester grades
{
disp=2;
printf("to know your marks type rollno,else type 0\n all semesters? y/n?\n");
scanf("%d %c",&r,&sem);
if(sem=='n')
{
printf("enter semester(1-5)\n");
scanf("%d",&sm);
}
if(r==0&&sem=='y')//all rno,all sem
{
sprintf(sql, "select rollno,sem,%s from cse_grades",gpa);
disp=3;
}
else if(r==0&&sem=='n')//all rno,s
sprintf(sql, "select rollno,%s from cse_grades where sem=%d",gpa,sm);
else if(r!=0&&sem=='y')//r,all sem
sprintf(sql, "select sem,%s from cse_grades where rollno = %d ",gpa,r);
else//r,s
{
sprintf(sql, "select %s from cse_grades where rollno = %d and sem=%d",gpa,r,sm);
disp=1;
}
break;
}

case 'f'://toppers in each semester
sprintf(sql, "select sem,rollno,sgpa from cse_grades where (sgpa,sem) in (select max(sgpa),sem from cse_grades group by sem)");//toppers of each sem
break;

case 'g'://branch topper
sprintf(sql, "select sem,name,cgpa from cse_grades where cgpa =(select max(cgpa) from cse_grades where sem=5)");
break;
disp=3;
case 'h'://students above 'g' gpa
{
printf("\nchoose one :  'cgpa'  or 'sgpa'\n");
scanf("%s",&gpa);
printf("enter g value :\n");
scanf("%f",&g);
disp=1;
if(g<0)
error=1;
else
sprintf(sql, "select distinct(rollno) from cse_grades where %s > %f",gpa,g);
}
break;
}
printf("%s",sql);
mysql_query(conn, sql);
res = mysql_use_result(conn);
while ((row = mysql_fetch_row(res)) != NULL)
{
printf("%s ", row[0]);
if(disp==2)
printf("%s ", row[1]);
if(disp==3)
printf("%s", row[2]);
printf("\n");
}
printf("continue? y/n?");
scanf("%c",&c);
}
mysql_close(conn);
}
return 0;
}


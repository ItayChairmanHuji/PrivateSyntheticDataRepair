SELECT DISTINCT t1.idx - 1 AS id1,
                t2.idx - 1 AS id2
FROM (SELECT *, row_number() OVER () AS idx FROM data) AS t1
         JOIN
         (SELECT *, row_number() OVER () AS idx FROM data) AS t2
         ON (<REPLACE_ME>)
             AND t1.idx != t2.idx;
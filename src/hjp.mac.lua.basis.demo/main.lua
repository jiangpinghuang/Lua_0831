-- Main function for demo in Lua.
table1 = {true, nil, true, false, nil, true, nil}
table2 = {true, false, nil, false, nil, true, nil}

a1,b1,c1,d1,e1,f1,g1 = unpack( table1, 1, table.maxn(table1)  )
print ("table1:",a1,b1,c1,d1,e1,f1,g1)

a2,b2,c2,d2,e2,f2,g2 = unpack( table2, 1, table.maxn(table2)  )
print ("table2:",a2,b2,c2,d2,e2,f2,g2)
  --local i2h             = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  
  function disp(x, y)
    print(y)(x)
  end
  
  function main()
  x =3
  y = 5
  disp(x, y)
  end
  
  main()
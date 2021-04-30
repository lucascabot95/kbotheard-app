Morris.Donut({
    element: 'graph',
    data: [
      {value: 60, label: 'NDR'},
      {value: 15, label: 'NPDR Leve'},
      {value: 10, label: 'NPDR Moderada'},
      {value: 5, label: 'NPDR Severa'},
      {value: 5, label: 'PDR'}
    ],
    formatter: function (x) { return x + "%"}
  }).on('click', function(i, row){
    console.log(i, row);
  });
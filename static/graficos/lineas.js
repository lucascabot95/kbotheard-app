new Morris.Line({
    // ID of the element in which to draw the chart.
    element: 'myfirstchart',
    // Chart data records -- each entry in this array corresponds to a point on
    // the chart.
    data: [
      { year: '2016', value: 20, value2: 10 },
      { year: '2017', value: 10, value2: 5 },
      { year: '2018', value: 5, value2:1 },
      { year: '2019', value: 5, value2:2},
      { year: '2020', value: 30, value2:18 }
    ],
    // The name of the data record attribute that contains x-values.
    xkey: 'year',
    // A list of names of data record attributes that contain y-values.
    ykeys: ['value', 'value2'],
    // Labels for the ykeys -- will be displayed when you hover over the
    // chart.
    labels: ['Pacientes', 'Reportes'],
    lineWidth: 1.5,
    lineColors: ['#C14D9F', '#2CB4AC'],
    resize: true,
  });
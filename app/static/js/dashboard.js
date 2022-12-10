$(document).ready(function(){

  function drawingGrapth(datax, datay) {
    const ctx = document.getElementById('chartAlgorithm');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: datax,
          datasets: [{
            label: 'Lượt sử dụng',
            data: datay,
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });


  } 

  $.ajax({
    url: '/chart/admin/0',
    type: 'post',
    success: function(e){
      drawingGrapth(e.algorithms, e.counts)
      // const ctx = document.getElementById('chartAlgorithm');
      // new Chart(ctx, {
      //   type: 'bar',
      //   data: {
      //     labels: e.algorithms,
      //     datasets: [{
      //       label: 'Lượt sử dụng',
      //       data: e.counts,
      //       borderWidth: 1
      //     }]
      //   },
      //   options: {
      //     scales: {
      //       y: {
      //         beginAtZero: true
      //       }
      //     }
      //   }
      // });
    }
  });

  $('select').on('change', function() {
    $.ajax({
      url: '/chart/admin/'+this.value,
      type: 'post',
      success: function(e){
        for (let i=1;i<=9;i++)
          if (!e.algorithms_id.includes(i)){
            e.algorithms.splice(i-1, 0, e.algorithms_info[1][i-1]);
            e.counts.splice(i-1, 0, 0);
          }
        $('#chartAlgorithm').remove()
        $('.chart-container').append(`<canvas id="chartAlgorithm"></canvas>`)
        drawingGrapth(e.algorithms, e.counts)
        // const ctx = document.getElementById('chartAlgorithm');
        // new Chart(ctx, {
        //   type: 'bar',
        //   data: {
        //     labels: e.algorithms,
        //     datasets: [{
        //       label: 'Lượt sử dụng',
        //       data: e.counts,
        //       borderWidth: 1
        //     }]
        //   },
        //   options: {
        //     scales: {
        //       y: {
        //         beginAtZero: true
        //       }
        //     }
        //   }
        // });
      }
    });
  });
});


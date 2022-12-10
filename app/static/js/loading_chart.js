$(document).ready(function(){
  // let href = window.location.href;
  // let url = new URL(href);
  // let param = url.pathname.substring(1).split("/")[1];

  $(".viewChart").click(function(){
    let param = $(this).attr("data-value")
    $.ajax({
      url: '/chart/'+param,
      type: 'post',
      beforeSend: function(){
      // Show image container
      // $("#loader").show();
      },
      success: function(e){
        let algorithms_array = e.algorithms;
        let algorithms = [];
        for (let index = 0; index < algorithms_array.length; index++) {
          let algorithm = {...algorithms_array[index]}
          algorithms.push(algorithm)
        }
        const labels = [];
        const points_rmse = [];
        const points_mae = [];
        const points_precision = [];
        const points_recall = [];
        const points_f1 = [];
        let is_binary = false;
        let points_array = e.points;
        for (let index = 0; index < points_array.length; index++) {
          let point = {...points_array[index]}
          let algorithm = algorithms.find(x => x[0] == point[0])
          labels.push(algorithm[1])
          if (algorithm[0] == 5 || algorithm[0] == 8){
            is_binary = true;
            points_precision.push(point[3]);
            points_recall.push(point[4]);
            points_f1.push(point[5]);
          }else{
            points_rmse.push(point[1])
            points_mae.push(point[2])
          }
        }
        let datasets = is_binary ? [
          {
            label: 'Precision',
            data: points_precision,
            backgroundColor: "#3e95cd",
            stack: 'Stack 0',
          },
          {
            label: 'Recall',
            data: points_recall,
            backgroundColor: "#8e5ea2",
            stack: 'Stack 1',
          },
          {
            label: 'F1',
            data: points_f1,
            backgroundColor: "#4bc0c0",
            stack: 'Stack 2',
          },
        ] : [
          {
            label: 'RMSE',
            data: points_rmse,
            backgroundColor: "#3e95cd",
            stack: 'Stack 0',
          },
          {
            label: 'MAE',
            data: points_mae,
            backgroundColor: "#8e5ea2",
            stack: 'Stack 1',
          },
        ]

        const data = {
          labels: labels,
          datasets: datasets
        };

        const config = {
          type: 'bar',
          data: data,
          options: {
            plugins: {
              title: {
                display: false,
                text: 'Sơ đồ minh hoạ'
              },
            },
            responsive: true,
            interaction: {
              intersect: false,
            },
            scales: {
              x: {
                stacked: true,
              },
              y: {
                stacked: true
              }
            }
          }
        };
        
        const myChart = new Chart(
          document.getElementById('chart'),
          config
        );

        $("#modalChart").modal('show');
      },
    });

    $('#modalChart').on('hide.bs.modal', function () {
      $('#chart').remove();
      $(".modal-body").append('<canvas id="chart"></canvas>');
    })
  });

  $("#reForm").submit(function(){
    // e.preventDefault();
    let href = window.location.href;
    let url = new URL(href);
    let param = url.pathname.substring(1).split("/")[1];
    $.ajax({
      url: '/results/'+param,
      type: 'POST',
      beforeSend: function(){
      // Show image container
      $("#loader").show();
      },
      // complete:function(data){
      // // Hide image container
      // $("#loader").hide();
      // }
   });
  });
});
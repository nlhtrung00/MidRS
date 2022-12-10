$(document).ready(function(){
   
  $("#upload").submit(function(){
    $.ajax({
      url: '/files',
      type: 'post',
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
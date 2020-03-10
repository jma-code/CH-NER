
var data_per = [];
var data_loc = [];
var data_org = [];

function predict(){
    sentence = document.getElementById('input').value;

    $.ajax({
                url: '/predict',
                type: 'post',
                contentType: 'application/json',
                data: JSON.stringify(sentence),
                success: (data) => {
                    eval('var item=' + data)
                    for(var i=0; i<item.PER.length; i++){
                        data_per[i] = item.PER[i];
                    }
                    for(var i=0; i<item.LOC.length; i++){
                        data_loc[i] = item.LOC[i];
                    }
                    for(var i=0; i<item.ORG.length; i++){
                        data_org[i] = item.ORG[i];
                    }
//                    console.log(data_per);
//                    console.log(data_loc);
//                    console.log(data_org);

                    }
    });
}

function clean(){
    window.location.reload(true);
}

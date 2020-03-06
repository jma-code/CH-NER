function predict(){
    sentence = document.getElementById('input').value;

    $.ajax({
                url: '/predict',
                type: 'post',
                contentType: 'application/json',
                data: JSON.stringify(sentence),
                success: (data) => {
                    eval('var item=' + data)
                    var str_per = 'PER: '
                    var str_loc = 'LOC: '
                    var str_org = 'ORG: '
                    for(var i=0; i<item.PER.length; i++){
                        str_per = str_per + item.PER[i] + ' '
                    }
                    for(var i=0; i<item.LOC.length; i++){
                        str_loc = str_loc + item.LOC[i] + ' '
                    }
                    for(var i=0; i<item.ORG.length; i++){
                        str_org = str_org + item.ORG[i] + ' '
                    }

                    $('#output').text(str_per + '\n' + str_loc + '\n' + str_org);
                    }
    });
}

function clean(){
    alert('delete')
    window.location.reload(true)
}
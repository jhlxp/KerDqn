#include <linux/mm.h>
#include <linux/module.h>
#include <linux/math64.h>
#include <net/tcp.h>
#include <linux/init.h>
#include <linux/types.h>
#include <net/sock.h>
#include <linux/netlink.h>

#define kerdqn_NETLINK_TEST     23
#define kerdqn_USER_PORT    8023
static const u32 kerdqn_min_rtt_win_sec __read_mostly = 10;
static const u32 kerdqn_max_thr_win_sec __read_mostly = 10;

extern struct net init_net;

static char kerdqn_netlink_kmsg[64];
static struct sock *kerdqn_nlsk = NULL;
static u32 kerdqn_cwnd = 10;

struct kerdqn_bictcp {
	u32 rtt_probe;
	u32 min_rtt_us; 
	u32 min_rtt_stamp;
	u32 max_bw_bps; 
	u32 max_bw_stamp;
	u32 last_update_stamp;
	u32 train_count;
	u32 before_probe_cwnd;
    u32 bw;
    u32 rtt;
	u32 recovery_count;
	u32 recovery_flag;
	u32 probe_rtt_list[3];
};

static void kerdqn_tcp_init(struct sock *sk)
{
	struct kerdqn_bictcp *qc = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	tp -> snd_ssthresh = TCP_INFINITE_SSTHRESH;
	qc -> rtt_probe = 0;
	qc -> min_rtt_us = 0xffffffff;
	qc -> min_rtt_stamp = tcp_jiffies32;
	qc -> max_bw_bps = 0;
	qc -> max_bw_stamp = tcp_jiffies32;
	qc -> last_update_stamp = tcp_jiffies32;
	qc -> train_count = 0;
	qc -> before_probe_cwnd = 10;
    qc -> bw = 0;
    qc -> rtt = 0;
	qc -> recovery_count = 0;
	qc -> recovery_flag = 0;
	qc -> probe_rtt_list[0] = 0xffffffff;
	qc -> probe_rtt_list[1] = 0xffffffff;
	qc -> probe_rtt_list[2] = 0xffffffff;
}

static u32 kerdqn_tcp_recalc_ssthresh(struct sock *sk)
{
	return TCP_INFINITE_SSTHRESH; 
}

static void kerdqn_get_minrtt_maxbw(struct sock *sk, const struct rate_sample *rs){

	struct kerdqn_bictcp *qc = inet_csk_ca(sk);
	bool filter_expired_minrtt;
	bool filter_expired_maxthr;
	u64 bw;

	if( rs -> rtt_us < 0 ){
		return ;
	}

	filter_expired_minrtt = after(tcp_jiffies32, qc -> min_rtt_stamp + kerdqn_min_rtt_win_sec * HZ);
	filter_expired_maxthr = after(tcp_jiffies32, qc -> max_bw_stamp + kerdqn_max_thr_win_sec * HZ);

	if( rs->interval_us < 0 ){
		return ;
	}

    bw = div64_long((u64)rs->delivered * (1 << 24), rs->interval_us);

    qc -> bw = (u32)bw;
    qc -> rtt = (u32)rs -> rtt_us;

	if (rs -> rtt_us > 0 &&
	    (rs -> rtt_us < qc -> min_rtt_us ||
	     (filter_expired_minrtt && !rs->is_ack_delayed))) {
		qc -> min_rtt_us = rs->rtt_us;		
		qc -> min_rtt_stamp = tcp_jiffies32;
	}

	if (rs -> rtt_us > 0 &&
	    (qc -> max_bw_bps < bw ||
	     (filter_expired_maxthr && !rs->is_ack_delayed))) {
		qc -> max_bw_bps = bw;
		qc -> max_bw_stamp = tcp_jiffies32;
	}

}

static int kerdqn_send_usrmsg(char *pbuf, uint16_t len)
{
    struct sk_buff *nl_skb;
    struct nlmsghdr *nlh;  

    int ret;

    nl_skb = nlmsg_new(len, GFP_ATOMIC);
    if(!nl_skb)
    {
        printk("netlink alloc failure\n");
        return -1;
    }

    nlh = nlmsg_put(nl_skb, 0, 0, kerdqn_NETLINK_TEST, len, 0);
    if(nlh == NULL)
    {
        printk("nlmsg_put failaure \n");
        nlmsg_free(nl_skb);  
        return -1;
    }

    memcpy(nlmsg_data(nlh), pbuf, len);
    ret = netlink_unicast(kerdqn_nlsk, nl_skb, kerdqn_USER_PORT, MSG_DONTWAIT);
    
    return ret;
}

static void kerdqn_send_bw_rtt(u32 bw,u32 rtt, u32 max_bw, u32 min_rtt,u32 cwnd){
    char *kmsg;
    snprintf(kerdqn_netlink_kmsg, sizeof(kerdqn_netlink_kmsg), "(%u;%u;%u;%u;%u;)", bw,rtt,max_bw,min_rtt,cwnd);
    kmsg = kerdqn_netlink_kmsg;
    kerdqn_send_usrmsg(kmsg, strlen(kmsg));
}

static void kerdqn_netlink_rcv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = NULL;
    char *umsg = NULL;
    u32 cwnd_receive = 0;

    if(skb->len >= nlmsg_total_size(0))
    {

        nlh = nlmsg_hdr(skb);  
        umsg = NLMSG_DATA(nlh); 
        if(umsg)
        {
			cwnd_receive = simple_strtol(umsg,NULL,10);
			printk("(umsg:%s)(cwnd_receive:%u)",umsg,cwnd_receive);
			kerdqn_cwnd = cwnd_receive;
        }
    }
}

static void kerdqn_tcp_cong_main(struct sock *sk, const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct kerdqn_bictcp *qc = inet_csk_ca(sk);
	bool training_timer_expired = after(tcp_jiffies32, qc -> last_update_stamp + msecs_to_jiffies(80)); 
	u32 count = 0;
	u32 min_value = 0;
	int i = 0;
	u32 min_cwnd = 10;

	printk("(kerdqn_get_minrtt_maxbw)");
	kerdqn_get_minrtt_maxbw(sk,rs);

    if(qc -> rtt_probe == 0){
        tp -> snd_cwnd = kerdqn_cwnd;
    }
	
	if(training_timer_expired){

		if(qc -> rtt_probe == 1){
			qc -> train_count--;
			qc -> probe_rtt_list[qc->train_count] = rs -> rtt_us;
			if(qc -> train_count == 0){
				qc -> rtt_probe = 0;
				tp -> snd_cwnd = qc -> before_probe_cwnd;
				qc -> recovery_count = 4;
				qc -> recovery_flag = 1;

				min_value = qc -> probe_rtt_list[0];
				for(i=0;i<count;i++){
					if(min_value>=qc->probe_rtt_list[i]){
						min_value = qc->probe_rtt_list[i];
					}
				}
				if(min_value > 0){
					qc -> min_rtt_us = min_value;
				}
				for(i=0;i<count;i++){
						qc -> probe_rtt_list[i] = 0xffffffff;
				}
			}
		}

		if(qc -> recovery_flag == 1){

			qc -> recovery_count--;
			if(qc -> recovery_count == 0){
				qc -> recovery_flag = 0;
			}
		}

		if(qc -> rtt_probe == 0){
			if(qc -> train_count > 95){
				qc -> rtt_probe = 1;
				qc -> min_rtt_stamp = tcp_jiffies32;
				qc -> train_count = 3;
				count = qc -> train_count;
				qc -> before_probe_cwnd = tp -> snd_cwnd;
				tp -> snd_cwnd = max(tp -> snd_cwnd/20, min_cwnd);
			}
			else{
				qc -> train_count++;
			}
		}

		if(qc -> rtt_probe == 0 && qc -> recovery_flag == 0){
			kerdqn_send_bw_rtt(qc->bw, qc->rtt,qc->max_bw_bps, qc->min_rtt_us, tp->snd_cwnd);
		}

		qc -> last_update_stamp = tcp_jiffies32;
	}

}

static void kerdqn_tcp_state(struct sock *sk, u8 new_state)
{
}

static void kerdqn_tcp_cwnd_event(struct sock *sk, enum tcp_ca_event event)
{
}

static void kerdqn_tcp_acked(struct sock *sk, const struct ack_sample *sample)
{
}

static u32 kerdqn_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

static struct netlink_kernel_cfg cfg = { 
        .input  = kerdqn_netlink_rcv_msg, 
};  


static struct tcp_congestion_ops kerdqn __read_mostly = {
	.init		= kerdqn_tcp_init,
	.ssthresh	= kerdqn_tcp_recalc_ssthresh,
	.cong_control	= kerdqn_tcp_cong_main,
	.set_state	= kerdqn_tcp_state,
	.undo_cwnd	= kerdqn_undo_cwnd,
	.cwnd_event	= kerdqn_tcp_cwnd_event,
	.pkts_acked     = kerdqn_tcp_acked,
	.owner		= THIS_MODULE,
	.name		= "kerdqn",
};

static int __init kerdqn_register(void)
{
	BUILD_BUG_ON(sizeof(struct kerdqn_bictcp) > ICSK_CA_PRIV_SIZE);

    kerdqn_nlsk = (struct sock *)netlink_kernel_create(&init_net, kerdqn_NETLINK_TEST, &cfg);
    if(kerdqn_nlsk == NULL)
    {   
        printk("netlink_kernel_create error !\n");
        return -1; 
    }   
    printk("netlink_test_init\n");

	return tcp_register_congestion_control(&kerdqn);
}

static void __exit kerdqn_unregister(void)
{
	tcp_unregister_congestion_control(&kerdqn);

    if (kerdqn_nlsk){
        netlink_kernel_release(kerdqn_nlsk); 
        kerdqn_nlsk = NULL;
    }   
    printk("netlink_test_exit!\n");
}

module_init(kerdqn_register);
module_exit(kerdqn_unregister);

MODULE_AUTHOR("xuheng");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("TCP kerdqn_");
MODULE_VERSION("2.3");

    